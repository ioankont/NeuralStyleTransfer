from Trainer import run_style_transfer
content_path="/content/drive/MyDrive/picturesStyleTransfer/dat/chicago.jpg"
style_path = "/content/drive/MyDrive/picturesStyleTransfer/dat/picaso.jpg"
best_result, best_loss = run_style_transfer(content_path,style_path, num_iterations=2000,content_weight=1e4,style_weight=7,tv_weight=10)
