"""Example: Deploy a model to HuggingFace Inference Endpoints.

Usage:
    python deploy_hf_model.py run

Requires:
    - HF_TOKEN environment variable
    - A model already pushed to HuggingFace Hub
"""

from metaflow import FlowSpec, step

from metaflow_extensions.serve import Deployment, ServiceSpec, endpoint, initialize


class MyModelService(ServiceSpec):
    @initialize(backend="huggingface", cpu=1, memory=2048)
    def init(self):
        self.model = self.artifacts.flow.model

    @endpoint
    def predict(self, request_dict):
        return {"prediction": self.model.predict(request_dict["input"])}


class DeployHFModelFlow(FlowSpec):
    @step
    def start(self):
        self.model = {"name": "my-model", "version": "1.0"}
        self.next(self.deploy)

    @step
    def deploy(self):
        self.deployment = (
            Deployment(
                MyModelService,
                step=self.start,
                config={
                    "repository": "user/my-model",
                    "instance_type": "nvidia-a10g.xlarge",
                    "region": "us-east-1",
                },
            )
            .audit("predict", payload={"input": [1, 2, 3]})
            .promote()
        )
        print(f"Deployed: {self.deployment.endpoint_url}")
        self.next(self.end)

    @step
    def end(self):
        print(f"Deployed version: {self.deployment.version}")
        print("Done!")


if __name__ == "__main__":
    DeployHFModelFlow()
