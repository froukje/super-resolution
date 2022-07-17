torch-model-archiver --model-name srnet --version 1.0 --model-file ../srresnet.py --serialized-file ../saved_models/best_model-epoch=223-val_loss=0.29.ckpt --handler srnet_handler.py --force
mv srnet.mar model-store
