import copy
import logging

import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
)
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, AdamW

from transformers_impl.data import *
from transformers_impl.utils import *


class FedTrainer:
    r""" Implements fine-tuning of transformer models in federated learning setup.
         Currently supports the following aggregation strategy for local model updates.
         - FedAdam (FedOpt)
         - FedProx
         - FedAvg
    """

    def __init__(
        self, args, tokenizer, model, local_data_idxs, train_data, val_data, test_data,
    ):
        self.args = args
        self.num_clients = max(int(args.C * args.K), 1)
        self.tokenizer = tokenizer
        self.model = model
        self.local_data_idxs = local_data_idxs
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        # initialize the server optimizer according to the aggregation algorithm
        self._init_opt()

    def _init_opt(self):
        # we don't need server optimizer in case of fedprox and fedavg + centralized training
        if self.args.K == 1 or self.args.algorithm != "fedopt":
            self.optimizer = None
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.server_lr)

    def train(self):
        best_f1_loss = float("-inf")
        patience = 0

        logging.info(f"Picking {self.num_clients} random clients per round.")

        for round_idx in range(1, self.args.rounds + 1):
            local_loss = []
            client_epoch_list = np.array([self.args.E] * self.num_clients)
            client_idxs = np.random.choice(range(self.args.K), self.num_clients, replace=False)

            # --------------------------------------------------------------------------------
            # Ideally, we should maintain n copies of local client updates in-memory and
            # aggregate the local client updates to obtain a global model. However, this is
            # computationally infeasible due to limited GPU memory. In order to avoid GPU OOM
            # issue, we accumulate the local client updates and normalize by number of clients.
            start = True
            for client_idx, epoch in zip(client_idxs, client_epoch_list):
                client = Client(
                    self.args,
                    epoch,
                    self.train_data,
                    self.local_data_idxs[client_idx],
                    self.tokenizer,
                )
                w, loss = client.train(model=copy.deepcopy(self.model))
                local_loss.append(loss)

                if start:
                    w_avg = copy.deepcopy(w)
                    start = False
                else:
                    w = copy.deepcopy(w)
                    for k in w_avg.keys():
                        w_avg[k] += w[k]

                del client, w
                torch.cuda.empty_cache()

            # normalizing the global weights
            for k in w_avg.keys():
                w_avg[k] = torch.div(w_avg[k], len(client_idxs))
            # --------------------------------------------------------------------------------

            if self.optimizer == None:
                # no optimization in case of fedprox and fedavg + centralized training
                self.model.load_state_dict(w_avg)
            else:
                # aggregate the local updates using server optimizer in case of fedopt
                self.optimizer.zero_grad()
                optimizer_state = self.optimizer.state_dict()
                # set model global grads
                new_model = copy.deepcopy(self.model)
                new_model.load_state_dict(w_avg)
                with torch.no_grad():
                    for parameter, new_parameter in zip(
                        self.model.parameters(), new_model.parameters()
                    ):
                        parameter.grad = parameter.data - new_parameter.data
                        # because we go to the opposite direction of the gradient
                model_state_dict = self.model.state_dict()
                new_model_state_dict = new_model.state_dict()
                for k in dict(self.model.named_parameters()).keys():
                    new_model_state_dict[k] = model_state_dict[k]
                self.model.load_state_dict(new_model_state_dict)
                # instantiate the optimizer and copy the optimizer state tracked above
                self._init_opt()
                self.optimizer.load_state_dict(optimizer_state)
                self.optimizer.step()
                del new_model  # to avoid GPU OOM issue

            avg_train_loss = sum(local_loss) / len(local_loss)

            # evaluate the model on validation set
            val_metrics, val_loss = self.eval(stage="valid")
            logging.info(
                f"Round: {round_idx}... \tAverage Train Loss: {round(avg_train_loss, 3)}... \tDev Loss: {round(val_loss, 3)}... "
                f"\tDev Accuracy: {val_metrics['valid/accuracy']}... \tAUC Score: {val_metrics['valid/auc']}... \tPrecision: {val_metrics['valid/precision']}... "
                f"\tRecall: {val_metrics['valid/recall']}... \tF1: {val_metrics['valid/f1_score']}... \tMCC: {val_metrics['valid/mcc']}"
            )

            test_metrics = {
                "test/accuracy": 0.0,
                "test/auc": 0.0,
                "test/precision": 0.0,
                "test/recall": 0.0,
                "test/f1_score": 0.0,
                "test/mcc": 0.0,
            }
            test_loss = 0.0

            # early stop if we don't improve till patience level
            if val_metrics["valid/f1_score"] > best_f1_loss:
                logging.info(
                    f"Dev f1 score improved ({best_f1_loss:.4f} -> {val_metrics['valid/f1_score']:.4f}). Saving model!"
                )
                best_f1_loss = val_metrics["valid/f1_score"]
                patience = 0
                # save the model and tokenizer
                if torch.cuda.device_count() > 1:
                    self.model.module.save_pretrained(self.args.save)
                else:
                    self.model.save_pretrained(self.args.save)
                self.tokenizer.save_pretrained(self.args.save)
            else:
                patience += 1
                logging.info(f"Early stopping counter {patience} out of {self.args.es_patience}")
                if patience == self.args.es_patience:
                    # load the model and tokenizer with best performance
                    if torch.cuda.device_count() > 1:
                        self.model.module.from_pretrained(self.args.save)
                    else:
                        self.model.from_pretrained(self.args.save)
                    self.tokenizer.from_pretrained(self.args.save)
                    # evaluate the model on test set
                    test_metrics, test_loss = self.eval(stage="test")
                    logging.info(
                        f"FINAL TESTING\n... \tTest Loss: {round(test_loss, 3)}... Test Accuracy: {test_metrics['test/accuracy']}...  "
                        f"\tAUC Score: {test_metrics['test/auc']}... \tPrecision: {test_metrics['test/precision']}... \tRecall: {test_metrics['test/recall']}... "
                        f"\tF1: {test_metrics['test/f1_score']}... \tMCC: {test_metrics['test/mcc']}"
                    )
                    break

            # finally evaluate the model on the test set
            if round_idx == self.args.rounds:
                # load the model and tokenizer with best performance
                if torch.cuda.device_count() > 1:
                    self.model.module.from_pretrained(self.args.save)
                else:
                    self.model.from_pretrained(self.args.save)
                self.tokenizer.from_pretrained(self.args.save)
                # evaluate the model on test set
                test_metrics, test_loss = self.eval(stage="test")
                logging.info(
                    f"FINAL TESTING\n... \tTest Loss: {round(test_loss, 3)}... Test Accuracy: {test_metrics['test/accuracy']}...  "
                    f"\tAUC Score: {test_metrics['test/auc']}... \tPrecision: {test_metrics['test/precision']}... \tRecall: {test_metrics['test/recall']}... "
                    f"\tF1: {test_metrics['test/f1_score']}... \tMCC: {test_metrics['test/mcc']}"
                )

        return self.model

    def eval(self, stage="valid"):
        assert stage in ("valid", "test"), f"stage: {stage} not supported"
        data = self.val_data if stage == "valid" else self.test_data
        dataloader = DataLoader(
            data,
            batch_size=self.args.batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(self.args.class_weights_array).cuda()
        )

        self.model.eval()
        pred_probs, labels = [], []

        batch_loss = []
        for batch in dataloader:
            # transfer the data (tensors) to GPU
            batch = {k: v.cuda() for k, v in batch.items()}

            with torch.no_grad():
                output = self.model(**batch)
                logits = output.logits
                loss = criterion(logits, batch["labels"])
                batch_loss.append(loss.item())

            # gather the predictions and labels which will be used later for calculating metrics
            probs = torch.softmax(logits, dim=-1)
            pred_probs.append(probs)
            labels.append(batch["labels"])

        epoch_loss = sum(batch_loss) / len(batch_loss)

        pred_probs = torch.cat(pred_probs)
        preds = torch.argmax(pred_probs, dim=-1)
        labels = torch.cat(labels)

        accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        if self.args.num_classes < 3:
            auc = roc_auc_score(
                labels.cpu().numpy(), pred_probs.cpu().numpy()[:, 1], labels=self.args.classes,
            )
        else:
            auc = roc_auc_score(
                labels.cpu().numpy(),
                pred_probs.cpu().numpy(),
                multi_class="ovo",
                labels=self.args.classes,
            )
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            labels.cpu().numpy(),
            preds.cpu().numpy(),
            average="weighted" if self.args.num_classes > 2 else "macro",
            labels=self.args.classes,
        )
        mcc = matthews_corrcoef(labels.cpu().numpy(), preds.cpu().numpy())

        metrics = {
            f"{stage}/accuracy": 100.0 * accuracy,
            f"{stage}/auc": auc,
            f"{stage}/precision": precision,
            f"{stage}/recall": recall,
            f"{stage}/f1_score": f1_score,
            f"{stage}/mcc": mcc,
        }

        return metrics, epoch_loss


class Client:
    def __init__(self, args, epochs, data, idxs, tokenizer):
        self.args = args
        self.epochs = epochs
        self.dataloader = DataLoader(
            CustomDataset(data, idxs),
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(tokenizer),
        )

    def train(self, model):
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(self.args.class_weights_array).cuda()
        )
        optimizer = AdamW(
            model.parameters(),
            lr=self.args.client_lr,
            weight_decay=0.01 if self.args.model_type == "fnet" else 0,
        )

        # use the weights of global model for proximal term calculation
        global_model = copy.deepcopy(model)

        model.train()

        epoch_loss = []
        for epoch in range(1, self.epochs + 1):
            batch_loss = []
            for batch in self.dataloader:
                # transfer the data (tensors) to GPU
                batch = {k: v.cuda() for k, v in batch.items()}

                # flush the gradients and perform a forward pass
                optimizer.zero_grad()
                output = model(**batch)
                logits = output.logits

                # calculate the loss
                # proximal term in fedprox acts as a kind of L2 regularization
                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                # don't include the proximal term for central training
                if int(self.args.K) == 1:
                    loss = criterion(logits, batch["labels"])
                else:
                    loss = criterion(logits, batch["labels"]) + (self.args.mu / 2) * proximal_term

                # perform the backward pass and update the local model parameters
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        total_loss = sum(epoch_loss) / len(epoch_loss)

        return model.state_dict(), total_loss
