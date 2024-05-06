..
  # SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  #
  # Licensed under the Apache License, Version 2.0 (the "License");
  # you may not use this file except in compliance with the License.
  # You may obtain a copy of the License at
  #
  # http://www.apache.org/licenses/LICENSE-2.0
  #
  # Unless required by applicable law or agreed to in writing, software
  # distributed under the License is distributed on an "AS IS" BASIS,
  # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  # See the License for the specific language governing permissions and
  # limitations under the License

.. _environment_variables:

Environment variables
=====================
 
Environment variables that can be set to control some global settings of the NVIDIA® nvImageCodec library.

PYNVIMGCODEC_VERBOSITY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Level of logging, when using the Python API. Possible values are

    1: Error
    2: Warning
    3: Info
    4: Debug
    5: Trace (only debug builds)

NVIMGCODEC_MAX_JPEG_SCANS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Maximum allowed number of progressive JPEG scans. Default value is 256. This limit is set to prevent a denial-of-service vulnerability via a crafted JPEG (see https://cure53.de/pentest-report_libjpeg-turbo.pdf)


