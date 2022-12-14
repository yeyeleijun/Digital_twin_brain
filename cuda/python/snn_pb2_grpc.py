# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import snn_pb2 as snn__pb2


class SnnStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Init = channel.unary_stream(
        '/snn.Snn/Init',
        request_serializer=snn__pb2.InitRequest.SerializeToString,
        response_deserializer=snn__pb2.InitResponse.FromString,
        )
    self.Run = channel.unary_stream(
        '/snn.Snn/Run',
        request_serializer=snn__pb2.RunRequest.SerializeToString,
        response_deserializer=snn__pb2.RunResponse.FromString,
        )
    self.Measure = channel.unary_stream(
        '/snn.Snn/Measure',
        request_serializer=snn__pb2.MetricRequest.SerializeToString,
        response_deserializer=snn__pb2.MetricResponse.FromString,
        )
    self.Updateprop = channel.stream_unary(
        '/snn.Snn/Updateprop',
        request_serializer=snn__pb2.UpdatePropRequest.SerializeToString,
        response_deserializer=snn__pb2.UpdatePropResponse.FromString,
        )
    self.Updategamma = channel.stream_unary(
        '/snn.Snn/Updategamma',
        request_serializer=snn__pb2.UpdateGammaRequest.SerializeToString,
        response_deserializer=snn__pb2.UpdateGammaResponse.FromString,
        )
    self.Updategammawithresult = channel.stream_stream(
        '/snn.Snn/Updategammawithresult',
        request_serializer=snn__pb2.UpdateGammaRequest.SerializeToString,
        response_deserializer=snn__pb2.UpdateGammaWithResultResponse.FromString,
        )
    self.Updatehyperpara = channel.stream_unary(
        '/snn.Snn/Updatehyperpara',
        request_serializer=snn__pb2.UpdateHyperParaRequest.SerializeToString,
        response_deserializer=snn__pb2.UpdateHyperParaResponse.FromString,
        )
    self.Updatesample = channel.stream_unary(
        '/snn.Snn/Updatesample',
        request_serializer=snn__pb2.UpdateSampleRequest.SerializeToString,
        response_deserializer=snn__pb2.UpdateSampleResponse.FromString,
        )
    self.Shutdown = channel.unary_unary(
        '/snn.Snn/Shutdown',
        request_serializer=snn__pb2.ShutdownRequest.SerializeToString,
        response_deserializer=snn__pb2.ShutdownResponse.FromString,
        )


class SnnServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Init(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Run(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Measure(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Updateprop(self, request_iterator, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Updategamma(self, request_iterator, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Updategammawithresult(self, request_iterator, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Updatehyperpara(self, request_iterator, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Updatesample(self, request_iterator, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Shutdown(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_SnnServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Init': grpc.unary_stream_rpc_method_handler(
          servicer.Init,
          request_deserializer=snn__pb2.InitRequest.FromString,
          response_serializer=snn__pb2.InitResponse.SerializeToString,
      ),
      'Run': grpc.unary_stream_rpc_method_handler(
          servicer.Run,
          request_deserializer=snn__pb2.RunRequest.FromString,
          response_serializer=snn__pb2.RunResponse.SerializeToString,
      ),
      'Measure': grpc.unary_stream_rpc_method_handler(
          servicer.Measure,
          request_deserializer=snn__pb2.MetricRequest.FromString,
          response_serializer=snn__pb2.MetricResponse.SerializeToString,
      ),
      'Updateprop': grpc.stream_unary_rpc_method_handler(
          servicer.Updateprop,
          request_deserializer=snn__pb2.UpdatePropRequest.FromString,
          response_serializer=snn__pb2.UpdatePropResponse.SerializeToString,
      ),
      'Updategamma': grpc.stream_unary_rpc_method_handler(
          servicer.Updategamma,
          request_deserializer=snn__pb2.UpdateGammaRequest.FromString,
          response_serializer=snn__pb2.UpdateGammaResponse.SerializeToString,
      ),
      'Updategammawithresult': grpc.stream_stream_rpc_method_handler(
          servicer.Updategammawithresult,
          request_deserializer=snn__pb2.UpdateGammaRequest.FromString,
          response_serializer=snn__pb2.UpdateGammaWithResultResponse.SerializeToString,
      ),
      'Updatehyperpara': grpc.stream_unary_rpc_method_handler(
          servicer.Updatehyperpara,
          request_deserializer=snn__pb2.UpdateHyperParaRequest.FromString,
          response_serializer=snn__pb2.UpdateHyperParaResponse.SerializeToString,
      ),
      'Updatesample': grpc.stream_unary_rpc_method_handler(
          servicer.Updatesample,
          request_deserializer=snn__pb2.UpdateSampleRequest.FromString,
          response_serializer=snn__pb2.UpdateSampleResponse.SerializeToString,
      ),
      'Shutdown': grpc.unary_unary_rpc_method_handler(
          servicer.Shutdown,
          request_deserializer=snn__pb2.ShutdownRequest.FromString,
          response_serializer=snn__pb2.ShutdownResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'snn.Snn', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
