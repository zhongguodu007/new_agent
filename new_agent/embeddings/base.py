
from pydantic import (
    BaseModel,
    Field,
    model_validator,
)
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader, 
    Docx2txtLoader
)

from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names
)
import os
os.environ['OPENAI_API_BASE'] = 'https://open.bigmodel.cn/api/paas/v4/'
os.environ["ZHIPUAI_API_KEY"] = 'df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc'
class EmbeddingModel(BaseModel, Embeddings):

    client: Any = None
    model: str = 'embedding-2'
    chunk_size: int = 10
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    class Cofig:

        extra = 'forbid'
        allow_population_by_field_name = True

    
    @model_validator(mode='before')
    def validate_enviroment(cls, values:Dict) ->Dict:
        zhipuai_api_key = get_from_dict_or_env(
            values, "zhipuai_api_key", "ZHIPUAI_API_KEY"
        )
        values["zhipuai_api_key"] = (
            convert_to_secret_str(zhipuai_api_key) if zhipuai_api_key else None
        )
        values["zhipuai_api_base"] = values["zhipuai_api_base"] or os.getenv(
            "OPENAI_API_BASE"
        )
        values["zhipuai_api_type"] = get_from_dict_or_env(
            values,
            "zhipuai_api_type",
            "OPENAI_API_TYPE",
            default="",
        )
        
        client_params = {
            "api_key": values["zhipuai_api_key"].get_secret_value() if values["zhipuai_api_base"] else None,
            "base_url": values["zhipuai_api_base"],
        }
        if not values.get("client"):
            values["client"] = OpenAI(
                **client_params
            ).embeddings

        return values
    
    @property
    def _invocation_params(self) -> Dict[str, Any]:

        params: Dict = {"model": self.model, **self.model_kwargs}
        return params
    

    def _get_len_safe(self, texts: List[str], chunk_size: Optional[int] = None
                      ) -> List[List[float]]:
        
        _chunk_size = chunk_size if chunk_size else self.chunk_size

        batch_embeddings: List[List[float]] = []

        for i in range(0, len(texts), _chunk_size):

            response = self.client.create(
                input=texts[i:i+_chunk_size], **self._invocation_params
            )
            if not isinstance(response, dict):
                response = response.dict()

            batch_embeddings.extend(d['embedding'] for d in response['data'])

        return batch_embeddings
        
    def embed_documents(self, texts: List[str], chunk_size:Optional[int]=None) ->List[List[float]]:
        '''
        texts：文本片段
        chunk_size：文本片段的最大长度

        return 返回文本片段的嵌入向量，以List[List[float]]
        '''

        return self._get_len_safe(texts, chunk_size=chunk_size)
    

    def embed_query(self, text: str) -> List[float]:

        '''
        text：文本

        return 返回文本的嵌入向量
        '''

        return self._get_len_safe(texts=[text])[0]


class TextSpliter:

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None
        ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    
    def _get_loader(self, file_name):
        _, ext = os.path.splitext(file_name)

        if ext == '.pdf':
            return PyPDFLoader(file_name)
        elif ext ==  '.txt':
            return TextLoader(file_name)
        elif ext == '.csv':
            return CSVLoader(file_name, source_column="text")
        elif ext == '.docx':
            return Docx2txtLoader(file_name)
        else:
            raise ValueError(f"不支持的文件格式：{ext}")
        
    @property
    def spliter(self):
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
    
    def split_text(self, file_name)-> List[str]:

        '''
        file_name：传入的文件名

        return 以List[str]格式返回文章片段
        '''

        documents = self._get_loader(file_name).load()
        texts = self.spliter.split_documents(documents)
        return [doc.page_content for doc in texts]


if __name__ == "__main__":
    # splitor = TextSpliter()

    # text = splitor.split_text('./其他有意思的论文/对目标检测的面向目标的关系蒸馏.pdf')

    # model = EmbeddingModel(zhipuai_api_key = 'df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc', zhipuai_api_base='https://open.bigmodel.cn/api/paas/v4/')
    # res = model.embed_documents(texts=text)
    # print(len(res))
    pass