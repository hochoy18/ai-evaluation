import os

from deepeval import assert_test
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.models.llms.deepseek_model import DeepSeekModel

from deepeval.models import OpenAIModel
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams
from rich import print as rprint
from dotenv import load_dotenv

load_dotenv()
model = OpenAIModel(
    model="qwen4-max-preview",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,
)

# model = DeepSeekModel()
def test_correctness():
    correctness_metric = GEval(
        name="Correctness",
        criteria="根据“预期输出”（expected output），判断“实际输出”（actual_output）是否正确",
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT],
        threshold=0.5,
        model=model
    )
    rprint(correctness_metric)
    rprint(correctness_metric.criteria)
    rprint(correctness_metric.evaluation_params)
    test_case = LLMTestCase(
        input="我持续咳嗽并伴有发烧，需要担心吗？?",
        # Replace this with the actual output from your LLM application
        actual_output="持续咳嗽和发烧可能是由病毒感染或其他更严重的疾病引起的。如果症状加重，或在几天内没有好转，请及时就医。",
        expected_output="持续咳嗽和发烧可能预示着多种疾病，从轻微的病毒感染，到肺炎或新冠肺炎（COVID-19）等更为严重的状况，都有可能。如果症状加重、持续数日不见好转，或者伴有呼吸困难、胸痛或其他令人担忧的体征，您应及时就医。"
    )
    rprint(test_case)
    assert_test(test_case, [correctness_metric])


"""
uv run deepeval test run src/deepeval_cases/test_example.py
或
uv run pytest src/deepeval_cases/test_example.py
"""