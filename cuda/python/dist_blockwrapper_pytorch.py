"""The Python implementation of the gRPC snn client."""
import grpc
from .snn_pb2 import *
from .snn_pb2_grpc import *
import numpy as np
import torch
import time
import pandas as pd
from queue import Queue
from multiprocessing.pool import ThreadPool as Pool
from collections import OrderedDict
from scipy import stats


class cache_property(object):
    def __init__(self, method):
        # record the unbound-method and the name
        self.method = method
        self.name = method.__name__
        self.__doc__ = method.__doc__

    def __get__(self, inst, cls):
        if inst is None:
            return self
        result = self.method(inst)
        setattr(inst, self.name, result)
        return result


class BlockWrapper:
    property_idx_trans = torch.tensor([19, -1, 0, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=torch.int64).cuda()
    MAX_MESSAGE_LENGTH = 2147483647
    buffersize = 1024 ** 3 // 4

    def __init__(self, address, path, delta_t, route_path=None, print_stat=False, force_rebase=False, overlap=1):
        self.print_stat = print_stat
        self._channel = grpc.insecure_channel(address,
            options = [('grpc.max_send_message_length', self.MAX_MESSAGE_LENGTH),
                       ('grpc.max_receive_message_length', self.MAX_MESSAGE_LENGTH)])
        self._stub = SnnStub(self._channel)
        if route_path is None:
            mode = InitRequest.CommMode.COMM_POINT_TO_POINT
            route_path = ""
        else:
            mode = InitRequest.CommMode.COMM_ROUTE_WITH_MERGE
        _init_resp = self._stub.Init(InitRequest(file_path=path,
                                                 route_path=route_path,
                                                 delta_t=delta_t,
                                                 mode=mode))

        block_id = []
        neurons_per_block = []
        subblk_info = []
        self._subblk_id_per_block = {}
        self.pool = Pool()

        subblk_base = 0
        cortical_subblk_start = 0 if overlap==1 else 2

        for i, resp in enumerate(_init_resp):
            assert (resp.status == SnnStatus.SNN_OK)
            block_id.append(resp.block_id)
            neurons_per_block.append(resp.neurons_per_block)
            for j, sinfo in enumerate(resp.subblk_info):
                if j == 0 and sinfo.subblk_id == cortical_subblk_start and len(subblk_info) > 0:
                    new_base = max([id for id, _ in subblk_info])
                    if force_rebase or new_base != cortical_subblk_start:
                        subblk_base = (new_base + overlap - 1) // overlap * overlap
                subblk_info.append((sinfo.subblk_id + subblk_base, sinfo.subblk_num))
            self._subblk_id_per_block[resp.block_id] = \
                (subblk_base, torch.unique(torch.tensor([sinfo.subblk_id + subblk_base for sinfo in resp.subblk_info], dtype=torch.int64).cuda()))
            assert self._subblk_id_per_block[resp.block_id][1].shape[0] > 0

        self._block_id = torch.tensor(block_id, dtype=torch.int64).cuda()
        self._neurons_per_block = torch.tensor(neurons_per_block, dtype=torch.int64).cuda()
        self._neurons_thrush = torch.cat([torch.tensor([0], dtype=torch.int64).cuda(),
                                          torch.cumsum(self._neurons_per_block, 0)])
        self._subblk_id = torch.tensor([s[0] for s in subblk_info], dtype=torch.int64).cuda()
        self._neurons_per_subblk = torch.tensor([s[1] for s in subblk_info], dtype=torch.int64).cuda()
        self._buff_len = self.buffersize // self._neurons_per_subblk.shape[0]
        self._subblk_id, _subblk_idx = torch.unique(self._subblk_id, return_inverse=True)
        if (_subblk_idx == torch.arange(_subblk_idx.shape[0], dtype=_subblk_idx.dtype, device=_subblk_idx.device)).all():
            self._subblk_idx = None
        else:
            self._subblk_idx = _subblk_idx

        self._reset_hyper_parameter()
        self._sample_order = None
        self._sample_num = 0
        self._iterations = 0

    def _reset_hyper_parameter(self):
        self._last_hyper_parameter = torch.ones([self._subblk_id.shape[0], 20], dtype=torch.float32).cuda()

    @property
    def total_neurons(self):
        return self._neurons_per_block.sum()

    @property
    def block_id(self):
        return self._block_id

    @property
    def subblk_id(self):
        return self._subblk_id

    @property
    def total_subblks(self):
        return self._subblk_id.shape[0]

    @cache_property
    def neurons_per_subblk(self):
        if self._subblk_idx is None:
            return self._neurons_per_subblk
        else:
            return self._reduceat(self._neurons_per_subblk)

    @property
    def neurons_per_block(self):
        return self._neurons_per_block

    def _reduceat(self, array):
        assert array.shape[-1] == self._subblk_idx.shape[0]
        out = torch.zeros(array.shape[:-1] + (self._subblk_id.shape[-1],), dtype=array.dtype, device=array.device)
        if len(array.shape) == 2:
            _subblk_idx = self._subblk_idx.unsqueeze(0).expand(array.shape[0], self._subblk_idx.shape[0])
        else:
            assert len(array.shape) == 1
            _subblk_idx = self._subblk_idx
        out.scatter_add_(-1, _subblk_idx, array)
        return out

    def last_time_stat(self):
        responses = self._stub.Measure(MetricRequest())
        rows = ["sending",
                "recving",
                "routing",
                "computing",
                "reporting",
                "copy_before_sending",
                "copy_after_recving",
                "parse_merge",
                "route_computing",
                "copy_before_reporting"]

        col = OrderedDict([('total_mean', lambda x: np.mean(x)),
                           ('total_std',  lambda x:  np.std(x)),
                           ('spatial_max_temporal_mean', lambda x: np.mean(np.max(x, axis=1), axis=0)),
                           ('spatial_argmax_temporal_mode', lambda x: stats.mode(np.argmax(x, axis=1), axis=None)[0]),
                           ('temporal_std_spatial_max', lambda x: np.max(np.std(x, axis=0), axis=0)),
                           ('temporal_std_spatial_argmax', lambda x: np.argmax(np.std(x, axis=0), axis=0))])

        name = []
        data = []
        for i, resps in enumerate(responses):
            d=[]
            for resp in resps.metric:
                if i == 0:
                    name.append(resp.name)
                d.append([getattr(resp,  row+"_duration") for row in rows])
            data.append(d)

        data = np.array(data)
        stat_data = [[f(data[:, :, i]) for f in col.values()] for i in range(len(rows))]
        table = pd.DataFrame(np.array(stat_data), index=pd.Index(rows), columns=list(col.keys()))
        return table

    def _merge_sbblk(self, array, weight=None):
        assert len(array.shape) in {1, 2} and array.shape[-1] == self._neurons_per_subblk.shape[0], \
            "error, {} vs {}".format(array.shape, self._neurons_per_subblk.shape)
        if self._subblk_idx is None:
            return array
        else:
            if weight is not None:
                array *= weight
            array = self._reduceat(array)
            if weight is not None:
                array /= self._reduceat(weight)
            assert array.shape[-1] == self._subblk_id.shape[0]
            return array

    def run(self, iterations, freqs=True, vmean=False, sample_for_show=False, imean=False, strategy=None):
        return_list = Queue()
        if strategy is None:
            strategy = RunRequest.Strategy.STRATEGY_SEND_PAIRWISE

        def _run():
            _recv_time = 0
            responses = self._stub.Run(RunRequest(iter=iterations,
                                                  iter_offset=self._iterations,
                                                  output_freq=freqs,
                                                  output_vmean=vmean,
                                                  output_sample=sample_for_show,
                                                  output_imean=imean,
                                                  strategy=strategy))
            for i in range(iterations):
                time1 = time.time()
                r = next(responses)
                assert (r.status == SnnStatus.SNN_OK)
                _recv_time += time.time() - time1
                j = i % self._buff_len
                #if (i % 800 == 0):
                #    print(i)
                if j == 0:
                    return_tuple = []
                    len = min(self._buff_len, iterations - i)
                    if freqs:
                        _freqs = np.empty([len, self._neurons_per_subblk.shape[0]], dtype=np.int32)
                        return_tuple.append(_freqs)
                    if vmean:
                        _vmeans = np.empty([len, self._neurons_per_subblk.shape[0]], dtype=np.float32)
                        return_tuple.append(_vmeans)
                    if sample_for_show:
                        _spike = np.empty([len, self._sample_num], dtype=np.uint8)
                        return_tuple.append(_spike)
                        _vi = np.empty([len, self._sample_num], dtype=np.float32)
                        return_tuple.append(_vi)
                    if imean:
                        _imean = np.empty([len, self._neurons_per_subblk.shape[0]], dtype=np.float32)
                        return_tuple.append(_imean)
                if freqs:
                    _freqs[j, :] = np.frombuffer(r.freq[0], dtype=np.int32)
                if vmean:
                    _vmeans[j, :] = np.frombuffer(r.vmean[0], dtype=np.float32)
                if sample_for_show:
                    _spike[j, :] = np.frombuffer(r.sample_spike[0], dtype=np.uint8)
                    _vi[j, :] = np.frombuffer(r.sample_vmemb[0], dtype=np.float32)
                if imean:
                    _imean[j, :] = np.frombuffer(r.imean[0], dtype=np.float32)
                if j == self._buff_len - 1 or i == iterations - 1:
                    return_list.put([torch.from_numpy(r) for r in return_tuple])
            return _recv_time

        def _error_callback(exc):
            print('error:', exc)
            raise ValueError

        process_thread = self.pool.apply_async(_run, error_callback=_error_callback)

        _run_time = 0
        for i in range(iterations):
            j = i % self._buff_len
            if j == 0:
                out = return_list.get()
                processed_out = list()
                time1 = time.time()
                if freqs:
                    _freqs = out.pop(0).cuda()
                    _freqs = self._merge_sbblk(_freqs)
                    processed_out.append(_freqs)
                if vmean:
                    _vmeans = out.pop(0).cuda()
                    _vmeans = self._merge_sbblk(_vmeans)
                    processed_out.append(_vmeans)
                if sample_for_show:
                    _spike = out.pop(0).cuda()
                    if self._sample_order is not None:
                        _spike = torch.index_select(_spike, -1, self._sample_order)
                    processed_out.append(_spike)
                    _vi = out.pop(0).cuda()
                    if self._sample_order is not None:
                        _vi = torch.index_select(_vi, -1, self._sample_order)
                    processed_out.append(_vi)
                if imean:
                    _imean = out.pop(0).cuda()
                    _imean = self._merge_sbblk(_imean)
                    processed_out.append(_imean)
                assert len(out) == 0
                _run_time += time.time() - time1

            if len(processed_out) == 1:
                yield (processed_out[0][j, :],)
            else:
                yield tuple(o[j, :] for o in processed_out)

        if self.print_stat:
            _recv_time = process_thread.get()
            print('run merge time: {}, recv time: {}'.format(_run_time, _recv_time))
            print(self.last_time_stat())
        else:
            process_thread.wait()

    @staticmethod
    def _lexsort(*args, **kwargs):
        for i in range(len(args)):
            idx = torch.argsort(args[i]) # the sort must be stable
            for a in args:
                a[:] = torch.take(a, idx)
            for v in kwargs.values():
                v[:] = v[idx].clone()

    @staticmethod
    def _histogram(number, thresh):
        idx = torch.bucketize(number, thresh, right=True) - 1
        idx, count = torch.unique(idx, return_counts=True)
        out = torch.zeros_like(thresh)[:-1]
        out[idx] = count
        return out

    def update_property(self, property_idx, property_weight, bid=None):
        if bid is not None:
            assert 0 <= bid and bid < len(self._neurons_per_block)

        assert isinstance(property_idx, torch.Tensor)
        assert isinstance(property_weight, torch.Tensor)

        assert property_weight.dtype == torch.float32
        assert property_idx.dtype == torch.int64
        assert len(property_idx.shape) == 2 and property_idx.shape[1] == 2
        assert len(property_weight.shape) == 1
        assert property_weight.shape[0] == property_idx.shape[0]

        time_1 = time.time()

        property_idx_0 = property_idx[:, 0].clone()
        property_idx_1 = torch.take(self.property_idx_trans, property_idx[:, 1])
        property_weight = property_weight.clone()
        del property_idx

        self._lexsort(property_idx_1, property_idx_0, value=property_weight)
        time_2 = time.time()
        prepare_time = time_2 - time_1
        process_time = 0

        def message_generator():
            nonlocal bid, prepare_time, process_time
            if bid is not None:
                yield UpdatePropRequest(block_id=bid,
                                        neuron_id=property_idx_0.cpu().numpy().astype(np.uint32).tolist(),
                                        prop_id=property_idx_1.cpu().numpy().astype(np.uint32).tolist(),
                                        prop_val=property_weight.cpu().numpy().astype(np.float32).tolist())
            else:
                time_3 = time.time()
                counts = self._histogram(property_idx_0, self._neurons_thrush)
                counts = torch.cat([torch.tensor([0], dtype=torch.int32).cuda(), torch.cumsum(counts, 0)])
                assert counts[-1] == property_idx_0.shape[0]
                time_4 = time.time()
                prepare_time += time_4 - time_3
                for bid in range(len(self._neurons_per_block)):
                    base = counts[bid]
                    thresh = counts[bid+1]
                    if base < thresh:
                        time_5 = time.time()
                        _property_idx_0 = (property_idx_0[base:thresh] - self._neurons_thrush[bid]).cpu().numpy().astype(np.uint32).tolist()
                        _property_idx_1 = property_idx_1[base:thresh].cpu().numpy().astype(np.uint32).tolist()
                        _property_weight = property_weight[base:thresh].cpu().numpy().astype(np.float32).tolist()
                        time_6 = time.time()
                        prepare_time += time_6 - time_5
                        out = UpdatePropRequest(block_id=bid,
                                                neuron_id=_property_idx_0,
                                                prop_id=_property_idx_1,
                                                prop_val=_property_weight)
                        yield out
                        time_7 = time.time()
                        process_time += time_7 - time_6

        response = self._stub.Updateprop(message_generator())
        if self.print_stat:
            print("Update Properties {}, prepare_time: {}, process_time: {}".format(response.success, prepare_time, process_time))
        self._reset_hyper_parameter()

    def _update_property_by_subblk(self, property_idx, property_hyper_parameter, process_hp, generate_request, grpc_method):
        assert isinstance(property_idx, torch.Tensor)
        assert isinstance(property_hyper_parameter, torch.Tensor)

        assert len(property_idx.shape) == 2 and property_idx.shape[1] == 2
        assert property_hyper_parameter.shape[0] == property_idx.shape[0]
        assert property_idx.dtype == torch.int64
        assert property_hyper_parameter.dtype == torch.float32
        time_1 = time.time()

        property_idx_0 = property_idx[:, 0].clone()
        property_idx_1 = torch.take(self.property_idx_trans, property_idx[:, 1])
        property_hyper_parameter = property_hyper_parameter.clone()
        del property_idx
        self._lexsort(property_idx_1, property_idx_0, value=property_hyper_parameter)

        sub_blk_idx_0 = torch.bucketize(property_idx_0, self._subblk_id, right=True) - 1
        assert (self._subblk_id[sub_blk_idx_0] == property_idx_0).all()

        process_hp(property_hyper_parameter, sub_blk_idx_0, property_idx_1)

        if self.print_stat:
            print("property_hyper_parameter", float(torch.max(property_hyper_parameter)),
                  float(torch.min(property_hyper_parameter)),
                  float(torch.mean(property_hyper_parameter)))

        time_2 = time.time()
        prepare_time = time_2 - time_1
        process_time = 0

        def message_generator():
            nonlocal prepare_time, process_time
            for bid in range(len(self._neurons_per_block)):
                time_3 = time.time()

                subblk_base, subblk_id = self._subblk_id_per_block[bid]
                idx = (property_idx_0 == subblk_id[torch.bucketize(property_idx_0, subblk_id, right=True) - 1]).nonzero(as_tuple=True)[0]
                time_4 = time.time()
                prepare_time += time_4 - time_3
                if idx.shape[0] != 0:
                    time_5 = time.time()
                    _property_idx_0 = (property_idx_0[idx] - subblk_base).cpu().numpy().astype(np.uint32).tolist()
                    _property_idx_1 = property_idx_1[idx].cpu().numpy().astype(np.uint32).tolist()
                    Request, hp_dict = generate_request(property_hyper_parameter[idx].cpu().numpy(),
                                                        bid, _property_idx_0, _property_idx_1)

                    out = Request(block_id=bid,
                                  brain_id=_property_idx_0,
                                  prop_id=_property_idx_1,
                                  **hp_dict)
                    time_6 = time.time()
                    prepare_time += time_6 - time_5
                    yield out
                    time_7 = time.time()
                    process_time += time_7 - time_6

        response = grpc_method(message_generator())
        if self.print_stat:
            print("Update Hyperparameter of Properties {}, prepare_time: {}, process_time: {}".format(response.success, prepare_time, process_time))

        return response

    def mul_property_by_subblk(self, property_idx, property_hyper_parameter, accumulate=False):
        def process_hp(property_hyper_parameter, sub_blk_idx_0, property_idx_1):
            assert len(property_hyper_parameter.shape) == 1
            # assert (property_hyper_parameter > 0).all()
            if not accumulate:
                property_hyper_parameter /= self._last_hyper_parameter[sub_blk_idx_0, property_idx_1]
            self._last_hyper_parameter[sub_blk_idx_0, property_idx_1] *= property_hyper_parameter

        def generate_request(property_hp, _1, _2, _3):
            _property_hp = property_hp.astype(np.float32).tolist()
            return UpdateHyperParaRequest, {"hpara_val": _property_hp}

        self._update_property_by_subblk(property_idx, property_hyper_parameter, process_hp, generate_request, self._stub.Updatehyperpara)

    def gamma_property_by_subblk(self, property_idx, gamma_concentration, gamma_rate, debug=False):
        assert len(gamma_rate.shape) == 1

        if gamma_concentration is None:
            gamma_concentration = torch.ones_like(gamma_rate)
        assert len(gamma_concentration.shape) == 1

        gamma_hp = torch.stack([gamma_concentration, gamma_rate], dim=1)

        def process_hp(gamma_hp, sub_blk_idx_0, property_idx_1):
            assert (gamma_hp >= 0).all()
            self._last_hyper_parameter[sub_blk_idx_0, property_idx_1] = 1

        debug_log = dict()

        def generate_request(gamma_hp, bid, _property_idx_0, _property_idx_1):
            _gamma_concentration = gamma_hp[:, 0].astype(np.float32).tolist()
            _gamma_rate = gamma_hp[:, 1].astype(np.float32).tolist()
            if debug:
                for brain_id, prop_id, gamma_1, gamma_2 in zip(_property_idx_0, _property_idx_1, _gamma_concentration, _gamma_rate):
                    debug_log[(bid, brain_id, prop_id)] = (gamma_1, gamma_2)
            return UpdateGammaRequest, {"gamma_concentration": _gamma_concentration,
                                        "gamma_rate": _gamma_rate}

        if not debug:
            self._update_property_by_subblk(property_idx, gamma_hp, process_hp, generate_request, self._stub.Updategamma)
        else:
            response = self._update_property_by_subblk(property_idx, gamma_hp, process_hp, generate_request, self._stub.Updategammawithresult)
            for r in response:
                brain_id = r.brain_id
                prop_id = r.prop_id
                bid = r.block_id
                if len(r.prop_val)>0:
                    val = np.frombuffer(r.prop_val[0], dtype=np.float32)
                    print("val", val.shape, val.max(), val.min(), val.mean(), val.std())
                else:
                    val = np.array([], dtype=np.float32)
                    print("val", val.shape)
                assert (bid, brain_id, prop_id) in debug_log
                alpha, beta = debug_log[(bid, brain_id, prop_id)]
                mean_error = abs(val.mean() - alpha/beta)/(alpha/beta)
                var_error = abs(val.var() - alpha/beta ** 2)/(alpha/beta ** 2)
                print("bid: {}, brain_id: {}, prop_id: {}, mean_err: {:.2f}, var_err: {:.2f}".format(bid, brain_id, prop_id, mean_error, var_error))

    def set_samples(self, sample_idx, bid=None):
        assert isinstance(sample_idx, torch.Tensor)
        assert len(sample_idx.shape) == 1
        assert (sample_idx.dtype == torch.int64)
        order, recover_order = torch.unique(sample_idx, return_inverse=True)
        if (order.shape[0] == sample_idx.shape[0]) and (order == sample_idx).all():
            self._sample_order = None
        else:
            self._sample_order = recover_order
        self._sample_num = order.shape[0]

        def message_generator():
            if bid is not None:
                yield UpdateSampleRequest(block_id=bid, sample_idx=order.cpu().numpy().astype(np.uint32).tolist())
            else:
                for i, _bid in enumerate(self._block_id):
                    idx = torch.logical_and(self._neurons_thrush[i] <= order, order < self._neurons_thrush[i+1])
                    _sample_idx = (order[idx] - self._neurons_thrush[i]).cpu().numpy().astype(np.uint32).tolist()
                    if len(_sample_idx) > 0:
                        yield UpdateSampleRequest(block_id=_bid, sample_idx=_sample_idx)

        response = self._stub.Updatesample(message_generator())
        if self.print_stat:
            print("Set Samples %s" % response.success)

    def close(self):
        self._channel.close()

    def shutdown(self):
        response = self._stub.Shutdown(ShutdownRequest())
        if self.print_stat:
            print("Shutdown GRPC Server %s" % response.shutdown)
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
