from src.models.ankh import AnkhModel
from src.models.esmc import ESMCModel
from src.models.prost_t5 import ProstT5Model
from src.models.prot_t5 import ProtT5Model
from src.models.protbert import ProtBERTModel

MODEL_REGISTRY = {
    "esmc": ESMCModel,
    "prot_t5": ProtT5Model,
    "prost_t5": ProstT5Model,
    "ankh": AnkhModel,
    "protbert": ProtBERTModel,
}
