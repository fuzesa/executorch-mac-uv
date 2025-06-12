"""Sample script to export a MobileNetV3 model to a bundled program for MPS backend using the Executorch framework."""

import logging
from pathlib import Path

import torch
from executorch import exir
from executorch.backends.apple.mps import MPSBackend
from executorch.devtools import BundledProgram
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir import EdgeCompileConfig, EdgeProgramManager, ExecutorchProgramManager
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.extension.export_util.utils import export_to_edge
from torchvision import models


def main() -> None:
    """Export the MobileNetV3 model to a bundled program."""
    main_logger = logging.getLogger(__name__)
    main_logger.setLevel(logging.INFO)

    model = models.mobilenet_v3_large()
    model.eval()

    example_inputs = (torch.randn(1, 3, 224, 224),)

    with torch.no_grad():
        model = torch.export.export_for_training(
            model,
            example_inputs,
            strict=True,
        ).module()
        edge: EdgeProgramManager = export_to_edge(
            model,
            example_inputs,
            edge_compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

    compile_specs = [CompileSpec("use_fp16", bytes(True))]

    lowered_module = to_backend(
        MPSBackend.__name__,
        edge.exported_program(),
        compile_specs,
    )

    executorch_program: ExecutorchProgramManager = export_to_edge(
        lowered_module,
        example_inputs,
        edge_compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=False))

    expected_output = model(*example_inputs)
    method_test_suites = [
        MethodTestSuite(
            method_name="forward",
            test_cases=[
                MethodTestCase(
                    inputs=example_inputs,
                    expected_outputs=[expected_output],
                ),
            ],
        ),
    ]
    main_logger.info("Expected output: %s", expected_output)

    bundled_program = BundledProgram(executorch_program, method_test_suites)
    bundled_program_buffer = serialize_from_bundled_program_to_flatbuffer(
        bundled_program,
    )

    model_name = "mv3_large_mps_fp16_bundled.pte"

    with Path.open(model_name, "wb") as file:
        file.write(bundled_program_buffer)
    main_logger.info("Saved bundled program to %s", model_name)


if __name__ == "__main__":
    main()
