import argparse, torch
from transformers import AutoTokenizer
from src.models.student import EmotionStudent
from src.models.onnx_exporter import export_onnx
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", default="outputs/onnx/student.onnx")
    args = ap.parse_args()
    import os; os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model = EmotionStudent(); model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall")
    qpath = export_onnx(model, tok, args.out)
    print(f"ONNX: {args.out}\nINT8: {qpath}")
if __name__ == "__main__": main()
