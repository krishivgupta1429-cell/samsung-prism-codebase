"""Jina/Qwen ONNX input compatibility shared by export and inference."""


def patch_position_ids(model):
    auto_model = model[0].auto_model
    if getattr(auto_model, "_prism_position_ids_patched", False):
        return
    original_forward = auto_model.forward

    def forward(input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        if position_ids is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            elif input_ids is not None:
                import torch

                position_ids = (
                    torch.arange(input_ids.shape[-1], device=input_ids.device)
                    .unsqueeze(0)
                    .expand_as(input_ids)
                )
            else:
                raise ValueError("position_ids requires input_ids or attention_mask")
        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

    auto_model.forward = forward
    auto_model._prism_position_ids_patched = True
