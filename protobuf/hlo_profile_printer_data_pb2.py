# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: hlo_profile_printer_data.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='hlo_profile_printer_data.proto',
  package='xla',
  syntax='proto3',
  serialized_options=b'\370\001\001',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x1ehlo_profile_printer_data.proto\x12\x03xla\"\xe5\x04\n\x15HloProfilePrinterData\x12H\n\x11\x63omputation_infos\x18\x01 \x03(\x0b\x32-.xla.HloProfilePrinterData.HloComputationInfo\x12\x1d\n\x15profile_counters_size\x18\x02 \x01(\x03\x12\x43\n\rextra_metrics\x18\x03 \x03(\x0b\x32,.xla.HloProfilePrinterData.ExtraMetricsEntry\x12\x19\n\x11\x65ntry_computation\x18\x04 \x01(\t\x1a\xc7\x01\n\x12HloInstructionInfo\x12\x11\n\tlong_name\x18\x01 \x01(\t\x12\x12\n\nshort_name\x18\x02 \x01(\t\x12\x10\n\x08\x63\x61tegory\x18\x03 \x01(\t\x12\x12\n\nflop_count\x18\x04 \x01(\x02\x12\x1c\n\x14transcendental_count\x18\x05 \x01(\x02\x12\x16\n\x0e\x62ytes_accessed\x18\x06 \x01(\x02\x12\x17\n\x0foptimal_seconds\x18\x07 \x01(\x02\x12\x15\n\rprofile_index\x18\x08 \x01(\x03\x1a\x83\x01\n\x12HloComputationInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x15\n\rprofile_index\x18\x02 \x01(\x03\x12H\n\x11instruction_infos\x18\x03 \x03(\x0b\x32-.xla.HloProfilePrinterData.HloInstructionInfo\x1a\x33\n\x11\x45xtraMetricsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x03:\x02\x38\x01\x42\x03\xf8\x01\x01\x62\x06proto3'
)




_HLOPROFILEPRINTERDATA_HLOINSTRUCTIONINFO = _descriptor.Descriptor(
  name='HloInstructionInfo',
  full_name='xla.HloProfilePrinterData.HloInstructionInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='long_name', full_name='xla.HloProfilePrinterData.HloInstructionInfo.long_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='short_name', full_name='xla.HloProfilePrinterData.HloInstructionInfo.short_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='category', full_name='xla.HloProfilePrinterData.HloInstructionInfo.category', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='flop_count', full_name='xla.HloProfilePrinterData.HloInstructionInfo.flop_count', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='transcendental_count', full_name='xla.HloProfilePrinterData.HloInstructionInfo.transcendental_count', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bytes_accessed', full_name='xla.HloProfilePrinterData.HloInstructionInfo.bytes_accessed', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='optimal_seconds', full_name='xla.HloProfilePrinterData.HloInstructionInfo.optimal_seconds', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='profile_index', full_name='xla.HloProfilePrinterData.HloInstructionInfo.profile_index', index=7,
      number=8, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=267,
  serialized_end=466,
)

_HLOPROFILEPRINTERDATA_HLOCOMPUTATIONINFO = _descriptor.Descriptor(
  name='HloComputationInfo',
  full_name='xla.HloProfilePrinterData.HloComputationInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='xla.HloProfilePrinterData.HloComputationInfo.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='profile_index', full_name='xla.HloProfilePrinterData.HloComputationInfo.profile_index', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='instruction_infos', full_name='xla.HloProfilePrinterData.HloComputationInfo.instruction_infos', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=469,
  serialized_end=600,
)

_HLOPROFILEPRINTERDATA_EXTRAMETRICSENTRY = _descriptor.Descriptor(
  name='ExtraMetricsEntry',
  full_name='xla.HloProfilePrinterData.ExtraMetricsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='xla.HloProfilePrinterData.ExtraMetricsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='xla.HloProfilePrinterData.ExtraMetricsEntry.value', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=602,
  serialized_end=653,
)

_HLOPROFILEPRINTERDATA = _descriptor.Descriptor(
  name='HloProfilePrinterData',
  full_name='xla.HloProfilePrinterData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='computation_infos', full_name='xla.HloProfilePrinterData.computation_infos', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='profile_counters_size', full_name='xla.HloProfilePrinterData.profile_counters_size', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='extra_metrics', full_name='xla.HloProfilePrinterData.extra_metrics', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='entry_computation', full_name='xla.HloProfilePrinterData.entry_computation', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_HLOPROFILEPRINTERDATA_HLOINSTRUCTIONINFO, _HLOPROFILEPRINTERDATA_HLOCOMPUTATIONINFO, _HLOPROFILEPRINTERDATA_EXTRAMETRICSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=40,
  serialized_end=653,
)

_HLOPROFILEPRINTERDATA_HLOINSTRUCTIONINFO.containing_type = _HLOPROFILEPRINTERDATA
_HLOPROFILEPRINTERDATA_HLOCOMPUTATIONINFO.fields_by_name['instruction_infos'].message_type = _HLOPROFILEPRINTERDATA_HLOINSTRUCTIONINFO
_HLOPROFILEPRINTERDATA_HLOCOMPUTATIONINFO.containing_type = _HLOPROFILEPRINTERDATA
_HLOPROFILEPRINTERDATA_EXTRAMETRICSENTRY.containing_type = _HLOPROFILEPRINTERDATA
_HLOPROFILEPRINTERDATA.fields_by_name['computation_infos'].message_type = _HLOPROFILEPRINTERDATA_HLOCOMPUTATIONINFO
_HLOPROFILEPRINTERDATA.fields_by_name['extra_metrics'].message_type = _HLOPROFILEPRINTERDATA_EXTRAMETRICSENTRY
DESCRIPTOR.message_types_by_name['HloProfilePrinterData'] = _HLOPROFILEPRINTERDATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

HloProfilePrinterData = _reflection.GeneratedProtocolMessageType('HloProfilePrinterData', (_message.Message,), {

  'HloInstructionInfo' : _reflection.GeneratedProtocolMessageType('HloInstructionInfo', (_message.Message,), {
    'DESCRIPTOR' : _HLOPROFILEPRINTERDATA_HLOINSTRUCTIONINFO,
    '__module__' : 'hlo_profile_printer_data_pb2'
    # @@protoc_insertion_point(class_scope:xla.HloProfilePrinterData.HloInstructionInfo)
    })
  ,

  'HloComputationInfo' : _reflection.GeneratedProtocolMessageType('HloComputationInfo', (_message.Message,), {
    'DESCRIPTOR' : _HLOPROFILEPRINTERDATA_HLOCOMPUTATIONINFO,
    '__module__' : 'hlo_profile_printer_data_pb2'
    # @@protoc_insertion_point(class_scope:xla.HloProfilePrinterData.HloComputationInfo)
    })
  ,

  'ExtraMetricsEntry' : _reflection.GeneratedProtocolMessageType('ExtraMetricsEntry', (_message.Message,), {
    'DESCRIPTOR' : _HLOPROFILEPRINTERDATA_EXTRAMETRICSENTRY,
    '__module__' : 'hlo_profile_printer_data_pb2'
    # @@protoc_insertion_point(class_scope:xla.HloProfilePrinterData.ExtraMetricsEntry)
    })
  ,
  'DESCRIPTOR' : _HLOPROFILEPRINTERDATA,
  '__module__' : 'hlo_profile_printer_data_pb2'
  # @@protoc_insertion_point(class_scope:xla.HloProfilePrinterData)
  })
_sym_db.RegisterMessage(HloProfilePrinterData)
_sym_db.RegisterMessage(HloProfilePrinterData.HloInstructionInfo)
_sym_db.RegisterMessage(HloProfilePrinterData.HloComputationInfo)
_sym_db.RegisterMessage(HloProfilePrinterData.ExtraMetricsEntry)


DESCRIPTOR._options = None
_HLOPROFILEPRINTERDATA_EXTRAMETRICSENTRY._options = None
# @@protoc_insertion_point(module_scope)
