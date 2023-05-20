import argparse
import os
import sys

absolute_path = os.path.abspath(__file__)
sys.path.append("/".join(os.path.dirname(absolute_path).split("/")[:-1]))

from src.pipeline.pipeline_builder import PipelineBuilder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--step', default="Test", type=str, help='step name')
    args = vars(parser.parse_args())

    pipeline = PipelineBuilder()
    pipeline.prepare_steps()

    if args["step"] and args["step"] != "All":
        pipeline.run_by_step_name(args["step"])
    else:
        pipeline.run_all()
