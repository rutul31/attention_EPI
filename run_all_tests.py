"""
Run inference on all 4 cell-line checkpoints using the HeLa test dataset.

Checkpoints:
  - L1_HELA.pt
  - L1_NU_GM12878.pt
  - L1_NU_HUVEC.pt
  - L1_NU_IMR90.pt

Note: The test dataset (nu_HeLa_combined_dataset.pt) was generated from HeLa-S3
cell line data. Evaluating other cell-line checkpoints on HeLa data gives a
cross-cell-line performance comparison.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from feature_extraction.seqgendataset import SeqGenDataset
from model.epintlm import EPIModel
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# All checkpoints to evaluate
checkpoints = {
    "HeLa-S3":   "./checkpoints/L1_HELA.pt",
    "GM12878":    "./checkpoints/L1_NU_GM12878.pt",
    "HUVEC":      "./checkpoints/L1_NU_HUVEC.pt",
    "IMR90":      "./checkpoints/L1_NU_IMR90.pt",
}

# Load test dataset (HeLa)
batch_size = 512
test_dataset = torch.load('./data/nu_HeLa_combined_dataset.pt', weights_only=False)

print(f"✅ Test dataset loaded: {len(test_dataset)} samples (HeLa-S3)\n")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def get_num_correct(preds, labels):
    predictions = (preds >= 0.5).float()
    return (predictions == labels).sum().item()

def evaluate(model, dataloader):
    """Evaluate model and return metrics."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    all_preds = torch.tensor([]).to(device)
    all_targets = torch.tensor([]).to(device)
    
    with torch.no_grad():
        for data in dataloader:
            enhancer_ids, promoter_ids, gene_data, labels = data
            enhancer_ids = enhancer_ids.to(device)
            promoter_ids = promoter_ids.to(device)
            gene_data = gene_data.to(device)
            labels = labels.to(device)
            
            outputs, _ = model(enhancer_ids, promoter_ids, gene_data)
            labels = labels.unsqueeze(1).float()
            all_targets = torch.cat((all_targets, labels.view(-1)))
            
            if labels.shape == torch.Size([1, 1]):
                labels = torch.reshape(labels, (1,))
            
            loss = model.criterion(outputs, labels)
            all_preds = torch.cat((all_preds, outputs.view(-1)))
            total_loss += loss.item()
            total_correct += get_num_correct(outputs, labels)
    
    pred_labels = (all_preds >= 0.5).float().cpu().numpy()
    true_labels = all_targets.cpu().numpy()
    
    aupr = average_precision_score(true_labels, all_preds.cpu().numpy())
    auc = roc_auc_score(true_labels, all_preds.cpu().numpy())
    f1 = f1_score(true_labels, pred_labels)
    acc = accuracy_score(true_labels, pred_labels)
    avg_loss = total_loss / len(dataloader)
    
    return {
        "loss": avg_loss,
        "auc": auc,
        "aupr": aupr,
        "f1": f1,
        "accuracy": acc,
        "positive_preds": int(np.sum(pred_labels)),
        "total_positives": int(np.sum(true_labels)),
    }


print("=" * 70)
print(f"{'Cell Line':<15} {'AUC':>8} {'AUPR':>8} {'F1':>8} {'Accuracy':>10} {'Loss':>10}")
print("=" * 70)

results = {}
for cell_line, ckpt_path in checkpoints.items():
    print(f"\n🔄 Loading checkpoint: {ckpt_path}")
    
    model = EPIModel().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('l1GRU.'):
            k = k.replace('l1GRU.', 'gru1.')
        elif k.startswith('l2GRU.'):
            k = k.replace('l2GRU.', 'gru2.')
        
        # skip embeddings as they have 4096 vs 4097 size mismatch
        if k in ['embedding_en.weight', 'embedding_pr.weight']:
            continue
            
        new_state_dict[k] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    metrics = evaluate(model, test_loader)
    results[cell_line] = metrics
    
    print(f"{'✅ ' + cell_line:<15} {metrics['auc']:>8.4f} {metrics['aupr']:>8.4f} "
          f"{metrics['f1']:>8.4f} {metrics['accuracy']:>10.4f} {metrics['loss']:>10.4f}")
    print(f"   Predicted positives: {metrics['positive_preds']}/{metrics['total_positives']}")
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n{'Cell Line':<15} {'AUC':>8} {'AUPR':>8} {'F1':>8} {'Accuracy':>10}")
print("-" * 50)
for cell_line, m in results.items():
    print(f"{cell_line:<15} {m['auc']:>8.4f} {m['aupr']:>8.4f} {m['f1']:>8.4f} {m['accuracy']:>10.4f}")

print(f"\n📊 All evaluations done on HeLa-S3 test set ({len(test_dataset)} samples)")
print("   Note: Non-HeLa checkpoints evaluated on HeLa data show cross-cell-line transfer.")
