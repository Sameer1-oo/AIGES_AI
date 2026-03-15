from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

_tokenizer = DistilBertTokenizer.from_pretrained("training/weights/text_model")
_model     = DistilBertForSequenceClassification.from_pretrained("training/weights/text_model")
_model.eval()

def detect_fake_text(text: str) -> dict:
    if not text or len(text.strip()) < 10:
        return {"result": "Too short", "confidence": 0.0}

    inputs = _tokenizer(text[:512], return_tensors="pt",
                        truncation=True, padding=True)
    with torch.no_grad():
        probs = torch.softmax(_model(**inputs).logits, dim=1)[0]

    ai_prob    = probs[1].item()
    human_prob = probs[0].item()

    if ai_prob > human_prob:
        return {"result": "AI-Generated",  "confidence": round(ai_prob * 100, 2)}
    else:
        return {"result": "Human-Written", "confidence": round(human_prob * 100, 2)}