import torch
import open_clip
import threading

class EVA02Model:
    _instance = None
    _lock = threading.Lock()
    
    MODEL_NAME = 'hf-hub:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k'
    PRETRAINED = 'laion2b_s4b_b131k'

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print('🔁 Loading EVA02 model...')
                    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
                        cls.MODEL_NAME, pretrained=cls.PRETRAINED
                    )
                    tokenizer = open_clip.get_tokenizer(cls.MODEL_NAME)
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = model.to(device).eval()
                    
                    cls._instance = {
                        'model': model,
                        'preprocess_val': preprocess_val,
                        'tokenizer': tokenizer,
                        'device': device
                    }
                    print('✅ EVA02 Model loaded.')
        return cls._instance
