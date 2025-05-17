#!/usr/bin/env python3
# coding: utf-8
# below codes are based and adapted from https://github.com/XinhaoMei/WavCaps/blob/master/retrieval/models/audio_encoder.py



import torch
import torch.nn as nn
from models.cnns import ResNet38, Cnn14
from models.htsat import HTSAT_Swin_Transformer
import dac
from pathlib import Path
from models.feature_extractor import AudioFeature, DACodes, DACLatents, chunk
from models.Codes_Embedding import LongCodesEmbedder
from models.Codes_htsat import Codec_Swin_Transformer
import audiotools as at #a helpful library from Descript for dealing with wavefiles
from encodecmae import load_model
from models.encodec_embedding import EncodecEmbedding

class AudioEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        if config["audio_encoder_args"]["type"] == "cnn":
            if config["audio_encoder_args"]["model"] == 'ResNet38':
                self.audio_enc = ResNet38(config)
            elif config["audio_encoder_args"]["model"] == 'Cnn14':
                self.audio_enc = Cnn14(config)

            if config["audio_encoder_args"]["pretrained"]:
                # loading pretrained CNN weights
                pretrained_cnn = torch.load('pretrained_models/audio_encoder/{}.pth'.
                                            format(config["audio_encoder_args"]["model"]))['model']
                dict_new = self.audio_enc.state_dict().copy()
                trained_list = [i for i in pretrained_cnn.keys()
                                if not ('fc' in i or i.startswith('spec') or i.startswith('logmel'))]
                for i in range(len(trained_list)):
                    dict_new[trained_list[i]] = pretrained_cnn[trained_list[i]]
                self.audio_enc.load_state_dict(dict_new)

            self.audio_width = 2048

        elif config["audio_encoder_args"]["type"] == "transformer":
            self.audio_enc = HTSAT_Swin_Transformer(
                spec_size=256,
                patch_size=4,
                patch_stride=(4, 4),
                num_classes=527,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[4, 8, 16, 32],
                window_size=8,
                config=config,
            )
            if config["audio_encoder_args"]["pretrained"]:
                audio_ckpt = torch.load("pretrained_models/audio_encoder/HTSAT.ckpt", map_location="cpu")["state_dict"]
                for key in list(audio_ckpt.keys()):
                    if key.startswith('sed_model') and ('spectrogram_extractor' not in key
                                                        and 'logmel_extractor' not in key):
                        v = audio_ckpt.pop(key)
                        audio_ckpt[key[10:]] = v
                self.audio_enc.load_state_dict(audio_ckpt, strict=False)
                param_names = [n for n, p in self.audio_enc.named_parameters()]
                # for n in param_names:
                #     print(n, "\t", "Loaded" if n in audio_ckpt else "Unloaded")
            self.audio_width = 768
            
         ### Add the implementation of Encodec feats  ###
        elif config["audio_encoder_args"]["type"] == "encodec":
            self.audio_enc = EncodecEmbedding(device=config["device"])
            self.audio_width = config["audio_encoder_args"]["audio_width"]  
            
        ### Add the implementation of ENCODECMAE  ###
        elif config["audio_encoder_args"]["type"] == "mae":
            # self.audio_enc = load_model('ec-ec-base_st', mode='train', device=config["device"])
            self.audio_enc = load_model('ec-ec-large', mode='train', device=config["device"])
            self.audio_enc.visible_encoder.compile=False
            # features = model.extract_features_from_array(wavs, layer=-1)
            self.audio_width = config["audio_encoder_args"]["audio_width"]  
            
        ### Add the implementation of VAMP for DAC ###
        elif config["audio_encoder_args"]["type"] == "vamp":
            import vampnet
            _codec = vampnet.load_codec()
            _model = vampnet.load_model(vampnet.DEFAULT_MODEL)
            self.interface = vampnet.interface.Interface(_codec, _model)
            # self.interface.codec = self.interface.codec.to("cpu")
            # self.interface.model = self.interface.model.to("cuda")
            # self.interface = self.interface.to( "cpu")
            # self.interface = self.interface.to("cuda" if torch.cuda.is_available() else "cpu")
            self.codes_enc = self.interface.codec#.to("cuda")
            self.audio_enc = self.interface.model
            self.audio_width = config["audio_encoder_args"]["audio_width"]  
        
        ### Add the implementation of dac_htsat
        elif config["audio_encoder_args"]["type"] == "dac_htsat":
            self.codes_enc = DACLatents(config)
            self.audio_enc = Codec_Swin_Transformer(
                spec_size=256,
                patch_size=4,
                patch_stride=(4, 4),
                num_classes=527,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[4, 8, 16, 32],
                window_size=8,
                config=config,
            )
            if config["audio_encoder_args"]["pretrained"]:
                audio_ckpt = torch.load("pretrained_models/audio_encoder/HTSAT.ckpt", map_location="cpu")["state_dict"]
                for key in list(audio_ckpt.keys()):
                    if key.startswith('sed_model') and ('spectrogram_extractor' not in key
                                                        and 'logmel_extractor' not in key):
                        v = audio_ckpt.pop(key)
                        audio_ckpt[key[10:]] = v
                self.audio_enc.load_state_dict(audio_ckpt, strict=False)
                param_names = [n for n, p in self.audio_enc.named_parameters()]
                # for n in param_names:
                #     print(n, "\t", "Loaded" if n in audio_ckpt else "Unloaded")
            self.audio_width = 768
            
        ### Add the implementation of mel features
        elif config["audio_encoder_args"]["type"] == 'mel':
            self.audio_enc = AudioFeature(config["audio_args"])
            self.audio_width = config["audio_encoder_args"]["audio_width"]  
            
        ### Add the implementation of DAC embedding
        elif config["audio_encoder_args"]["type"] == 'dac_embedder':
            self.codes_enc = DACodes(config)
            self.audio_enc = LongCodesEmbedder(config)
            self.audio_width = config["audio_encoder_args"]["audio_width"]  

        ### Add the implementation of DAC 
        elif config["audio_encoder_args"]["type"] == "dac":
            # self.audio_enc = DACodes(config)
            # device = torch.device('cuda')            
            model_path = Path(config["audio_encoder_args"]["dac_path"])
            # model_path = Path('/home/dhuang/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth')
            self.audio_enc = dac.DAC.load(model_path)
            # self.audio_enc = self.audio_enc.to(device)
            # self.audio_enc.eval()
            self.audio_width = config["audio_encoder_args"]["audio_width"]            
            
        else:
            raise NotImplementedError('No such audio encoder network.')

        if config["audio_encoder_args"]["freeze"]:
            if config["audio_encoder_args"]["type"] == "encodec":
                for name, param in self.audio_enc.model.named_parameters():
                    param.requires_grad = False
                    # print("self.audio_enc.named_parameters():  ", name, param.requires_grad)
            else:
                for name, param in self.audio_enc.named_parameters():
                    param.requires_grad = False
                    # print("self.audio_enc.named_parameters():  ", name, param.requires_grad)


    def forward(self, inputs):
        """

        :param inputs: audio features
        :return: encoded audio embeddings
        """
        if self.config["audio_encoder_args"]["type"] == "encodec":
            x = inputs[:,None,:].repeat(1, 2, 1).detach().clone()
            # print("x shape is: ", x.shape)
            # print(f"self.training in eval mode: {not self.training}")
            # print(f"self.training.audio_enc in eval mode: {not self.audio_enc.training}")
            with torch.no_grad():
                # print("N_CODEBOOKS: ", self.config["audio_encoder_args"]["N_CODEBOOKS"])
                latents = self.audio_enc.embedding(x)
                # print("latents shape is: ", torch.flatten(latents.permute(1, 0, 2, 3), start_dim=1).shape)
                # print("codes shape is: ", codes.shape)
                # print("torch.flatten(codes, start_dim=1): ", torch.flatten(codes, start_dim=1).shape)
            return torch.flatten(latents.permute(1, 0, 2, 3), start_dim=1)
        elif self.config["audio_encoder_args"]["type"] == "dac":
            # self.audio_enc.eval()
            # signals = AudioSignal(inputs)
            x = self.audio_enc.preprocess(inputs, self.config["audio_args"]["sr"])
            # x = torch.tensor(x[:,None,:])
            x = x[:,None,:].detach().clone()
            # print("inputs wav hz is: ", len(inputs[0]) / self.config["audio_args"]["max_length"])
            # print("x shape is: ", x.shape)
            # time.sleep(10)
            # print(f"self.training in eval mode: {not self.training}")
            # print(f"self.training.audio_enc in eval mode: {not self.audio_enc.training}")
            with torch.no_grad():
                # print("N_CODEBOOKS: ", self.config["audio_encoder_args"]["N_CODEBOOKS"])
                z, codes, latents, _, _ = self.audio_enc.encode(x, self.config["audio_encoder_args"]["N_CODEBOOKS"])
                # print("codes shape is: ", codes.shape)
                # print("torch.flatten(codes, start_dim=1): ", torch.flatten(codes, start_dim=1).shape)
            return torch.flatten(latents, start_dim=1).to(torch.float)
        elif self.config["audio_encoder_args"]["type"] == "vamp":
            inputs = [at.AudioSignal(input,  self.config["audio_args"]["sr"]) for input in inputs]
            inputs_batch = at.AudioSignal.batch(inputs)
            # print("inputs_batch shape: ", inputs_batch.shape)
            # tokens = self.interface.encode(inputs_batch)
            inputs_batch = self.interface.preprocess(inputs_batch).to(self.config["audio_encoder_args"]["DAC_Device"])
            # print("inputs_batch shape: ", inputs_batch.shape)
            self.codes_enc = self.codes_enc.to(self.config["audio_encoder_args"]["DAC_Device"])
            self.codes_enc.eval()
            with torch.no_grad():
                if self.config["device"] == "cpu":
                    tokens = self.codes_enc.encode(inputs_batch.samples, None)[1].detach().cpu()
                else:
                    tokens = self.codes_enc.encode(inputs_batch.samples, None)[1]
            mask = torch.zeros_like(tokens)
            # mask = torch.ones_like(tokens) # it will mask all with the same
            # print("mask shape: ", mask.shape)
            # TODO MODIFY THE FUNC HERE; vamp_em to be splited;
            chunked_seq = chunk(tokens, mask, self.audio_enc.n_codebooks, self.audio_enc.max_seq_len)
            # [print("chunk shape: ", chunk.shape, "mask_chunk shape: ", mask_chunk.shape) for chunk, mask_chunk in chunked_seq]
            # print("mask_chunk shape: ", chunked_seq[1].shape)
            # gen_fn = gen_fn or self.model.generate_em
            c_vamp_chunks = [
                self.audio_enc.generate_em(
                    codec=self.codes_enc.to(self.config["device"]),
                    time_steps=chunk.shape[-1],
                    start_tokens=chunk,
                    mask=mask_chunk,
                    sampling_temperature=1.0,
                    typical_filtering=True,
                    top_p=0.9,
                    sample_cutoff=1.0,
                    # **kwargs,
                )
                # for chunk, mask_chunk in tqdm(zip(z_chunks, mask_chunks), desc="vamping chunks")
                for chunk, mask_chunk in chunked_seq
            ]
            # concatenate the chunks
            c_vamp = torch.cat(c_vamp_chunks, dim=-1)
     
            return torch.flatten(c_vamp, start_dim=1)
        elif self.config["audio_encoder_args"]["type"] == "dac_embedder":
            codes = self.codes_enc(inputs)
            audio_encoded = self.audio_enc(codes)
            # return torch.flatten(audio_encoded, start_dim=1)
            return audio_encoded
        elif self.config["audio_encoder_args"]["type"] == "mae":
            # if self.config["device"] == "cuda":
            audio_encoded = self.audio_enc.extract_features_from_array(inputs, return_type='torch', layer=-1)
            # 使用全局平均池化
            if self.config["audio_encoder_args"]["dim_reduced"]:
                global_avg_pool = nn.AdaptiveAvgPool1d(768)
                audio_encoded_flat = torch.flatten(audio_encoded, start_dim=1)
                audio_encoded_reduced = global_avg_pool(audio_encoded_flat.unsqueeze(1)).squeeze(1)
                return audio_encoded_reduced
            else:
                return torch.flatten(audio_encoded, start_dim=1)
        elif self.config["audio_encoder_args"]["type"] == "mel":
            audio_encoded = self.audio_enc(inputs)
            return torch.flatten(audio_encoded, start_dim=1)
        elif self.config["audio_encoder_args"]["type"] == "dac_htsat":
            self.codes_enc = self.codes_enc.to("cuda")
            self.codes_enc.eval()
            with torch.no_grad():
                latents = self.codes_enc(inputs.to("cuda")) # [batch, n_codebooks * 8, frames]
            _b, _nc, _f = latents.shape
            if _nc == 64:
                if self.config["device"] == "cpu":
                    latents = latents[:,None,:,:].detach().cpu().clone()
                else:
                    latents = latents[:,None,:,:].detach().clone()
                # print("The latents shape is: ", latents.shape)
                audio_encoded = self.audio_enc(latents)
                # print("The audio_encoded shape is: ", audio_encoded.shape)
                return audio_encoded  
            else:
                #TODO pad and reshape the latents into shape of [batch, 1, *inferred, n_feats] 
                n_feats = self.audio_enc.spec_size // (self.audio_enc.spec_size // self.config["audio_args"]["n_mels"])
                sp_len = _nc*_f
                n_inf = (sp_len - sp_len%n_feats) // n_feats
                if self.config["device"] == "cpu":
                    latents = torch.flatten(latents, start_dim=1)[:, 0:n_inf*n_feats].detach().cpu().clone().reshape(_b, 1, n_inf, n_feats)
                else:
                    latents = torch.flatten(latents, start_dim=1)[:, 0:n_inf*n_feats].detach().clone().reshape(_b, 1, n_inf, n_feats)
                # latents = latents[:,None,:,:].detach().clone().reshape(_b, 1, n_inf, n_feats)
                audio_encoded = self.audio_enc(latents)
                # return torch.flatten(audio_encoded, start_dim=1)
                return audio_encoded
        
        else:
            audio_encoded = self.audio_enc(inputs)
            return audio_encoded
