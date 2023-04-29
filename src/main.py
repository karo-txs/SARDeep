from src.pipeline.pipeline_builder import PipelineBuilder

if __name__ == "__main__":
    pipeline = PipelineBuilder()
    pipeline.prepare_steps()
    pipeline.run()
