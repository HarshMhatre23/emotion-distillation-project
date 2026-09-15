import torch
from onnxruntime.quantization import quantize_dynamic, QuantType
def export_onnx(model, tokenizer, output_path):
    model.eval()
    dummy = tokenizer("I am happy.", return_tensors="pt")
    torch.onnx.export(model, (dummy["input_ids"], dummy["attention_mask"]), output_path,
        input_names=["input_ids","attention_mask"], output_names=["logits","dimensions"],
        dynamic_axes={"input_ids":{0:"batch",1:"seq"},"attention_mask":{0:"batch",1:"seq"},"logits":{0:"batch"},"dimensions":{0:"batch"}},
        opset_version=14)
    qpath = output_path.replace(".onnx", "_int8.onnx")
    quantize_dynamic(output_path, qpath, weight_type=QuantType.QInt8)
    return qpath
