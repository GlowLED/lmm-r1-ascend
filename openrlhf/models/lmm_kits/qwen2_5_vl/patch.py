from ..base.patch import BasePatch
import torch

class Qwen2_5_VLPatch(BasePatch):
    def _add_get_inputs_embeds():
        from transformers import Qwen2_5_VLForConditionalGeneration
        def get_inputs_embeds(self, input_ids, image_grid_thw=None, video_grid_thw=None, pixel_values=None, pixel_values_videos=None, **kwargs):
            # Qwen2_5_VLModel may place embed_tokens under .language_model or directly
            if hasattr(self.model, 'embed_tokens'):
                embed_tokens = self.model.embed_tokens
            else:
                embed_tokens = self.model.language_model.embed_tokens
            inputs_embeds = embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            else:
                fake_pixel_values = torch.zeros((16, 1176),device=inputs_embeds.device,dtype=self.visual.dtype)
                fake_image_grid_thw = torch.tensor([[1, 4, 4]],device=inputs_embeds.device,dtype=torch.int32)
                image_embeds = self.visual(fake_pixel_values, grid_thw=fake_image_grid_thw)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds + 0*image_embeds.mean()

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            return inputs_embeds

        Qwen2_5_VLForConditionalGeneration.get_inputs_embeds = get_inputs_embeds

    def _add_get_position_ids():
        from transformers import Qwen2_5_VLForConditionalGeneration
        def get_position_ids(self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None, packing=False, **kwargs):
            # get_rope_index may live on self, self.model, or not exist at all
            rope_fn = getattr(self, 'get_rope_index', None) or getattr(self.model, 'get_rope_index', None)
            if rope_fn is not None:
                position_ids, mrope_position_deltas = rope_fn(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                )
            else:
                # Fallback: build simple incremental position_ids and broadcast to 3 dims for mrope
                import torch
                seq_len = input_ids.shape[-1]
                pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
                position_ids = pos.unsqueeze(0).expand(3, -1, -1)  # [3, bs, seq_len]
                mrope_position_deltas = None
            if packing:
                position_ids = position_ids.permute(1,2,0) # [3,bs,seq_len] -> [bs,seq_len,3]
            return position_ids
        Qwen2_5_VLForConditionalGeneration.get_position_ids = get_position_ids

    def _add_offset_split_position_ids():
        from transformers import Qwen2_5_VLForConditionalGeneration
        def offset_split_position_ids(self,position_ids,hacked_position_ids):
            # This function is only called when using packing, in which case, the position_ids is permuted as [bs,seq_len,3]
            position_ids = position_ids.permute(2,0,1) # [bs,seq_len,3] -> [3,bs,seq_len]
            new_position_ids = position_ids.clone()
            for i in range(hacked_position_ids.size(0)):
                seq_idxes = torch.nonzero(hacked_position_ids[i]==0)[:,0]
                seq_idxes = torch.cat([seq_idxes, torch.tensor([hacked_position_ids.size(1)],device=seq_idxes.device)], dim=0)
                st = 0
                for seq_idx in seq_idxes:
                    if st == 0 and seq_idx == 0:
                        continue
                    #shape: [3,bs,seq_len]
                    raw_seq_position_ids = position_ids[:,i,st:seq_idx]
                    new_position_ids[:,i,st:seq_idx] = raw_seq_position_ids - raw_seq_position_ids[:,:1] + hacked_position_ids[i,st]
                    st = seq_idx
            return new_position_ids
        Qwen2_5_VLForConditionalGeneration.offset_split_position_ids = offset_split_position_ids
    
    def apply_liger_kernel():
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
        apply_liger_kernel_to_qwen2_5_vl()

    @classmethod
    def _load_all_patches(cls):
        cls._add_get_inputs_embeds()
        cls._add_get_position_ids()
        cls._add_offset_split_position_ids()

Patch = Qwen2_5_VLPatch()