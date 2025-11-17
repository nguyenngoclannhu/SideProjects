import pytest
from deepeval.models import OllamaModel
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric, SummarizationMetric
from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig
import time 
from CustomModel import CustomModel
import pandas as pd

# proxy_client = get_proxy_client("gen-ai-hub")
model = OllamaModel(
    model= "llama3.1",
    base_url="http://localhost:11434",
    temperature=0
)

qna_prompt = '''You are an expert in BIM architectural designs and Obayashi corporation's project regulations and standards. Use the following context to answer the question accurately. 
**Context**
{context}

**Question**: {query}

**Answer format**:
Provide a detailed answer based on the context above. If the context does not contain sufficient information to answer the question, respond with "The answer is not available in the provided context."
**Note**: 
- Do not invent any information not in the context gien.
- If the answer is not present in the context, respond with "The answer is not available in the provided context."'''

def get_prompt(context, query, is_summarize: bool = True):
    return qna_prompt.format(context=context, query=query)


# def get_odata_prompt(date, supplier_data, payment_term_data, purchashing_org_data, query):
#     return oData_prompt_template % (date, supplier_data, payment_term_data, purchashing_org_data, query)

chatbot = CustomModel()
is_async = False
answer_relevant_metric = AnswerRelevancyMetric(model = model, async_mode=is_async)
faithfullness_metric = FaithfulnessMetric(model = model, async_mode=is_async)
contextual_precision_metric = ContextualPrecisionMetric(model = model, async_mode=is_async)
contextual_recall_metric = ContextualRecallMetric(model = model, async_mode=is_async)
contextual_relevancy_metric = ContextualRelevancyMetric(model = model, async_mode=is_async)
summarization_metric = SummarizationMetric(model = model, async_mode=is_async,
        assessment_questions=[
        "Does the summary capture the core arguments or main points of the original text?",
        "Is the information presented in the summary factually consistent with the original text?",
        "Does the summary cover the main points without introducing inaccuracies or omissions?",
        "Does a higher score mean a more comprehensive summary?"
    ]
)

class TestCase():
    def __init__(self, query, output, chatbot):
        self.query = query
        self.expected_output = output
        self.chatbot = chatbot
        self.context = ""
    
    def get_prompt(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    def create_llm_test_cases(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    # def extract_context_chunks(self):
    #     return extract_chunks(self.context["results"][0]["results"][0]["dataRepository"]["documents"]) if "results" in self.context else [""]
    
    def _create_llm_test_cases(self, prompt, actual_output, context):
        return LLMTestCase(
            name = self.query,
            input = prompt,
            actual_output=actual_output,
            expected_output=self.expected_output,
            retrieval_context=context
        )

class QATestCases(TestCase):
    def __init__(self, query, output, context, chatbot=chatbot):
        super().__init__(query, output, chatbot)
        self.context = context
        print(f"Processing query: {query}, expected Output: {output}")
        # self.context = chatbot.search_documents(query)
        
    def get_prompt(self):
        return qna_prompt.format(context=self.context, query=self.query)
    
    def create_llm_test_cases(self):
        return self._create_llm_test_cases(
            self.get_prompt(),
            self.chatbot.chat(self.query)['answer'],
            [self.context])

    def get_context(self):
        return chatbot.search_documents(self.query)
    
    def search_documents(self):
        return chatbot.search_documents(self.query)
        
qa_test_cases = []
file_name = "20251111_Query(Sheet1).csv"

#read csv file
pd.read_csv(file_name).apply(
    lambda row: qa_test_cases.append(
        QATestCases(
            query=row['Query'],
            output=row['Expected Output'],
            context=row['Conext']
        )
    ), axis=1
)

# start = time.time()
# test_cases = [tc.create_llm_test_cases() for tc in qa_test_cases[:1]]
# print(">>> Time taken to create test cases: ", time.time() - start)

# evaluate(test_cases, [answer_relevant_metric, faithfullness_metric, 
# contextual_precision_metric, contextual_precision_metric], 
#          async_config=AsyncConfig(run_async=is_async, max_concurrent=5),
#          cache_config=CacheConfig(write_cache=False))

print(qa_test_cases[0].search_documents())