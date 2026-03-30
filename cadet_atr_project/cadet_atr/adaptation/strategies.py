# ─────────────────────────────────────────────────────────────
# PATCH FILE — strategies.py fixes
# Drop these methods into RealDataFinetuner to replace the originals.
# ─────────────────────────────────────────────────────────────
#
# BUG FIXED: double forward pass in _train_epoch
# -----------------------------------------------
# Original (broken):
#
#   loss = self.criterion(self.model(x), y)   ← first forward
#   loss.backward()
#   ...
#   correct += (self.model(x).argmax(1) == y).sum().item()  ← SECOND forward!
#
# This is wasteful (doubles GPU time per batch) and potentially
# wrong if dropout/batchnorm are active between the two calls.
# Fix: store logits once, reuse.

    def finetune(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        mode:         str = "head_only",
    ) -> nn.Module:
        """
        Fine-tune the model on real IR training data.
        Returns the adapted model with best validation accuracy.
        """
        print(f"\n[Finetune] Mode: {mode} | "
              f"train={len(train_loader.dataset)} real images")

        if mode == "head_only":
            self._freeze_all()
            self._unfreeze_head()
            lr = cfg.finetune_lr

        elif mode == "full":
            self._unfreeze_all()
            lr = cfg.finetune_lr_full

        elif mode == "layer_wise":
            self._freeze_all()
            self._unfreeze_head()
            lr = cfg.finetune_lr

        else:
            raise ValueError(f"Unknown fine-tune mode: {mode}")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        best_val_acc = 0.0
        no_improve   = 0

        for epoch in range(1, self.epochs + 1):

            if mode == "layer_wise" and epoch % 5 == 0:
                self._unfreeze_next_block(epoch)
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=lr * 0.1, weight_decay=cfg.weight_decay
                )

            train_loss, train_acc = self._train_epoch(train_loader, optimizer)
            scheduler.step()

            val_acc = self._quick_eval(val_loader)
            print(f"  Epoch {epoch:2d} | val_acc={val_acc:.4f} | "
                  f"train_acc={train_acc:.4f} | train_loss={train_loss:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve   = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"  ✓ Saved best → {self.save_path}")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"  Early stop at epoch {epoch}")
                    break

        self.model.load_state_dict(
            torch.load(self.save_path, map_location=self.device)
        )
        print(f"\n[Finetune] Done. Best val_acc={best_val_acc:.4f}")
        return self.model

    def _train_epoch(
        self,
        loader:    DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> tuple:
        """
        Single training epoch.

        FIX: logits are computed ONCE per batch and reused for both
        loss calculation and accuracy tracking.  The original code
        called self.model(x) a second time inside the accuracy line,
        doubling the forward-pass compute and potentially giving
        inconsistent results when dropout is active.
        """
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in tqdm(loader, desc="  finetune", leave=False):
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)

            optimizer.zero_grad()

            # ── Single forward pass — reuse logits for both loss and acc ──
            logits = self.model(x)
            loss   = self.criterion(logits, y)
            loss.backward()
            optimizer.step()

            # No second self.model(x) here — use the logits we already have
            total_loss += loss.item()
            correct    += (logits.detach().argmax(1) == y).sum().item()
            total      += y.size(0)

        return total_loss / len(loader), correct / total
