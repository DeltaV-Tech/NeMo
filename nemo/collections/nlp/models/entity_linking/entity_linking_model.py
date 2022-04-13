# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from transformers import AutoTokenizer

from nemo.collections.common.losses import MultiSimilarityLoss
from nemo.collections.nlp.data import EntityLinkingDataset
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import ChannelType, LogitsType, MaskType, NeuralType
from nemo.utils import logging

__all__ = ['EntityLinkingModel']


class EntityLinkingModel(NLPModel, Exportable):
    """
    Second stage pretraining of BERT based language model
    for entity linking task. An implementation of Liu et. al's
    NAACL 2021 paper Self-Alignment Pretraining for Biomedical Entity Representations.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"logits": NeuralType(('B', 'D'), LogitsType())}

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Initializes the SAP-BERT model for entity linking."""

        # tokenizer needed before super().__init__() so dataset and loader can process data
        self._setup_tokenizer(cfg.tokenizer)

        super().__init__(cfg=cfg, trainer=trainer)

        self.model = get_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=cfg.language_model.config_file,
            config_dict=cfg.language_model.config,
            checkpoint_file=cfg.language_model.lm_checkpoint,
        )

        # Token to use for the self-alignment loss, typically the first token, [CLS]
        self._idx_conditioned_on = 0
        self.loss = MultiSimilarityLoss()

    def _setup_tokenizer(self, cfg: DictConfig):
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_name, vocab_file=cfg.vocab_file, do_lower_case=cfg.do_lower_case
        )

        self.tokenizer = tokenizer

    @typecheck()
    def forward(self, input_ids, token_type_ids, attention_mask):
        hidden_states = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # normalize to unit sphere
        logits = torch.nn.functional.normalize(hidden_states[:, self._idx_conditioned_on], p=2, dim=1)
        return logits

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        input_ids, token_type_ids, attention_mask, concept_ids = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        train_loss = self.loss(logits=logits, labels=concept_ids)

        # No hard examples found in batch,
        # shouldn't use this batch to update model weights
        if train_loss == 0:
        #    train_loss = None
        #    lr = None
            logging.info("train_loss == 0 in EntityLinkingModel.training_step")
        #    val_loss = None
            logging.info(train_loss)
            logging.info(train_loss.size())
            logging.info(train_loss.ndim)
            logging.info(train_loss.numel())
            logging.info(train_loss.stride())
            logging.info(train_loss.element_size())
            logging.info(train_loss.type())


        #else:
        lr = self._optimizer.param_groups[0]["lr"]
        self.log("train_loss", train_loss)
        self.log("lr", lr, prog_bar=True)

        return {"loss": train_loss, "lr": lr}

#    def training_step_end(self, training_step_outputs):
#        logging.info("training_step_end called")
#        logging.info(training_step_outputs)
#        if torch.distributed.is_initialized():
#            tensor_list = [torch.zeros([], dtype=training_step_outputs["loss"].dtype,
#                device=training_step_outputs["loss"].device) for _ in range(3)]
#            logging.info(tensor_list)
#            torch.distributed.all_gather(tensor_list, training_step_outputs["loss"])
#            logging.info(tensor_list)
#            gb_count = 0
#            for i in range(len(tensor_list)):
#                if tensor_list[i] == 0:
#                    logging.info(f"got zero loss for rank {i} - returning")
#                    return 
#            #logging.info("reducing lr")
#            #all_lr = training_step_outputs["lr"]
#            #torch.distributed.all_reduce(all_lr, op=torch.distributed.ReduceOp.AVG)
#            #logging.info(all_lr)
#
#            out = {"loss": avg_loss, "lr": training_step_outputs["lr"]}
#            logging.info("distributed out")
#            logging.info(out)
#        else:
#            out = training_step_outputs
#
#        #logging.info(training_step_outputs)
#        return out

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, concept_ids = batch
        with torch.no_grad():
            logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
            val_loss = self.loss(logits=logits, labels=concept_ids)

        # No hard examples found in batch,
        # val loss not used to update model weights
        if val_loss == 0:
            logging.info("val_loss == 0 in EntityLinkingModel.validation_step")
        #    val_loss = None
            logging.info(val_loss)
            logging.info(val_loss.size())
            logging.info(val_loss.ndim)
            logging.info(val_loss.numel())
            logging.info(val_loss.stride())
            logging.info(val_loss.element_size())
            logging.info(val_loss.type())

        #else:
        

        self.log("val_loss", val_loss)
        logging.info(f"val loss: {val_loss}")

        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.

        Args:
            outputs: list of individual outputs of each validation step.
        Returns:
            
        """
        if outputs:
            avg_loss = torch.stack([x["val_loss"] for x in outputs if x["val_loss"] != None]).mean()
            self.log(f"val_loss", avg_loss, prog_bar=True)

            return {"val_loss": avg_loss}

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.data_file:
            logging.info(
                f"Dataloader config or file_path or processed data path for the train dataset is missing, \
                        so no data loader for train is created!"
            )

            self._train_dl = None
            return

        self._train_dl = self.setup_dataloader(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.data_file:
            logging.info(
                f"Dataloader config or file_path or processed data path for the val dataset is missing, \
                        so no data loader for validation is created!"
            )

            self._validation_dl = None
            return

        self._validation_dl = self.setup_dataloader(cfg=val_data_config)

    def setup_dataloader(self, cfg: Dict, is_index_data: bool = False) -> 'torch.utils.data.DataLoader':

        dataset = EntityLinkingDataset(
            tokenizer=self.tokenizer,
            data_file=cfg.data_file,
            max_seq_length=cfg.max_seq_length,
            is_index_data=is_index_data,
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=cfg.get("shuffle", True),
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

    @classmethod
    def from_pretrained(cls, name: str):
        pass
