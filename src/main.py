from src.pipeline.pipeline_builder import PipelineBuilder
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--step', default=None, type=str, help='step name')
    args = vars(parser.parse_args())

    pipeline = PipelineBuilder()
    pipeline.prepare_steps()

    if args["step"]:
        pipeline.run_by_step_name(args["step"])
    else:
        pipeline.run_all()
