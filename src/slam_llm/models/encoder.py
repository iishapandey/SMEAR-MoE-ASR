import os
import json
import types
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from jiwer import wer


class WhisperWrappedEncoder:
    @classmethod
    def load(cls, model_config):
        
        def extract_variable_length_features(self, x: torch.Tensor):
            """
            x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
                the mel spectrogram of the audio
            """
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1)

            # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
            # x = (x + self.positional_embedding).to(x.dtype)
            x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

            for block in self.blocks:
                x = block(x)

            x = self.ln_post(x)
            return x

        if model_config.encoder_path_hf is not None:
            from transformers import WhisperModel
            encoder = WhisperModel.from_pretrained(model_config.encoder_path_hf).encoder
        elif model_config.encoder_path.endswith('.pt'):
            import whisper
            encoder = whisper.load_model(name=model_config.encoder_path, device='cpu').encoder
            encoder.extract_variable_length_features = types.MethodType(extract_variable_length_features, encoder)
        else:
            print(f"Picking IndicWhisper from {model_config.encoder_path} dir")
            model_dir = model_config.encoder_path
            model = WhisperForConditionalGeneration.from_pretrained(model_dir)
            encoder= model.model.encoder
            print("IndicWhisper model loaded!!!")
        return encoder



# class IndicConformerEncoder:
#     @classmethod


class IndicWhisperWrappedEncoder:
    @classmethod
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.encoder = model.model.encoder

        # Move model to the correct device
        # self.model.to(device)
        print("IndicWhisper model loaded and ready!!!")

    def load(cls, model_config):
        print(f"Picking IndicWhisper from {model_config.encoder_path} dir")
        model_dir = model_config.encoder_path
        processor = WhisperProcessor.from_pretrained(model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        print("IndicWhisper model loaded!!!")
        return cls(model, processor)


    def get_decoder_states(self, filename: str, save_states: bool = True) -> str:
            """Transcribe audio and extract decoder hidden states."""
            audio, rate = torchaudio.load(filename)
            if rate != 16000:
                audio = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)(audio)

            inputs = self.processor(audio.squeeze(0), sampling_rate=16000, return_tensors="pt")
            inputs.input_features = inputs.input_features.to(dtype=model.dtype)

            # forced_decoder_ids = processor.get_decoder_prompt_ids(language="hi", task="transcribe")
            self.model.generation_config.forced_decoder_ids = None

            # 2. Call generate(), passing the new prompt IDs
            outputs = self.model.generate(
                inputs.input_features,
                language=lang,
                # forced_decoder_ids=forced_decoder_ids,  # Use the newly generated IDs
                output_hidden_states=True,
                return_dict_in_generate=True
            )

            # # The rest of your code remains the same
            last_token_hidden_states = outputs.decoder_hidden_states[-1]
            # last_decoder_state = last_token_hidden_states[-1]

            audio_name = filename.split("/")[-1].split(".")[0]
            if save_states:
                torch.save(last_token_hidden_states.cpu(), f"{self.whisper_rep_path}/{audio_name}_decoder_state.pt")

            # transcription = self.processor.batch_decode(
            #     generated.sequences, skip_special_tokens=True
            #     )[0]
            return last_token_hidden_states



class BEATsEncoder:
    @classmethod
    def load(cls, model_config):
        from .BEATs.BEATs import BEATs, BEATsConfig
        checkpoint = torch.load(model_config.encoder_path)
        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])

        return BEATs_model


@dataclass
class UserDirModule:
    user_dir: str
    
class EATEncoder:
    
    @classmethod
    def load(cls, model_config):
        import fairseq
        model_path = UserDirModule(model_config.encoder_fairseq_dir)
        fairseq.utils.import_user_module(model_path)
        EATEncoder, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        EATEncoder = EATEncoder[0]

        return EATEncoder
    
    def extract_features(self, source, padding_mask):
        return self.model.extract_features(source, padding_mask = padding_mask, mask=False, remove_extra_tokens = False)['x']

class CLAPEncoder: 

    @classmethod
    def load(cls, model_config): 
        from .CLAP.ase_model import ASE
        import ruamel.yaml as yaml
        with open(model_config.clap_config, 'r') as f: 
            clap_config = yaml.safe_load(f)
        clap_config['pd_text_support'] = model_config.get("pd_text_support", None)
        model = ASE(clap_config)
        checkpoint = torch.load(model_config.encoder_path)['model']
        model.load_state_dict(checkpoint)
        return model
    
class SpatialASTEncoder:
    @classmethod
    def load(cls, model_config):
        from functools import partial
        from .SpatialAST import SpatialAST 
        binaural_encoder = SpatialAST.BinauralEncoder(
            num_classes=355, drop_path_rate=0.1, num_cls_tokens=3,
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, 
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        checkpoint = torch.load(model_config.encoder_ckpt, map_location='cpu')
        binaural_encoder.load_state_dict(checkpoint['model'], strict=False) 
        return binaural_encoder

class WavLMEncoder(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    @classmethod
    def load(cls, model_config):
        from .wavlm.WavLM import WavLM, WavLMConfig
        checkpoint = torch.load(model_config.encoder_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        WavLM_model = WavLM(cfg)
        WavLM_model.load_state_dict(checkpoint['model'])
        assert model_config.normalize == cfg.normalize, "normalize flag in config and model checkpoint do not match"
 
        return cls(cfg, WavLM_model)

    def extract_features(self, source, padding_mask):
        return self.model.extract_features(source, padding_mask)[0]

class AVHubertEncoder:

    @classmethod
    def load(cls, model_config):
        import fairseq
        from .avhubert import hubert_pretraining, hubert, hubert_asr
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        model = models[0]
        return model

class HubertEncoder:
    @classmethod
    def load(cls, model_config):
        import fairseq
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        model = models[0]
        if model_config.encoder_type == "pretrain":
            pass
        elif model_config.encoder_type == "finetune":
            model.w2v_encoder.proj = None
            model.w2v_encoder.apply_mask = False
        else:
            assert model_config.encoder_type in ["pretrain", "finetune"], "input_type must be one of [pretrain, finetune]" 
        return model

class HfTextEncoder:

    @classmethod
    def load(cls, model_config):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_config.encoder_path)
        return model

class MusicFMEncoder(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    @classmethod
    def load(cls, model_config):
        from .musicfm.model.musicfm_25hz import MusicFM25Hz
        model = MusicFM25Hz(
            stat_path = model_config.encoder_stat_path,
            model_path = model_config.encoder_path,
            w2v2_config_path = model_config.get('encoder_config_path', "facebook/wav2vec2-conformer-rope-large-960h-ft")
        )
        return cls(model_config, model)

    def extract_features(self, source, padding_mask=None):
        _, hidden_states = self.model.get_predictions(source)
        out = hidden_states[self.config.encoder_layer_idx]
        return out

class Emotion2vecEncoder:

    @classmethod
    def load(cls, model_config):
        import fairseq
        model_path = UserDirModule(model_config.encoder_fairseq_dir)
        fairseq.utils.import_user_module(model_path)
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        model = model[0]

        return model


class IndicWhisper:
    def __init__(self, model_dir: str, lang: str = "hi", cuda_device: str = "0"):
        """
        Initialize IndicWhisper ASR model.

        Args:
            model_dir (str): Path to Whisper model directory.
            lang (str): Language code.
        """
        # Make only the selected GPU visible
        # os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        print(f"[INFO] Loading IndicWhisper model from: {model_dir}")
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir)
        self.lang = lang

    def get_decoder_states(self, filename: str, save_states: bool = True) -> str:
        """Transcribe audio and extract decoder hidden states."""
        audio, rate = torchaudio.load(filename)
        if rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)(audio)

        inputs = self.processor(
            audio.squeeze(0), sampling_rate=16000, return_tensors="pt"
        ).to(self.device)

        # Step 1: Generate tokens
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=self.lang)
        generated = self.model.generate(
            inputs.input_features,
            forced_decoder_ids=forced_decoder_ids,
            return_dict_in_generate=True,
        )

        # Step 2: Forward pass with hidden states
        with torch.no_grad():
            outputs = self.model(
                inputs.input_features,
                decoder_input_ids=generated.sequences,
                output_hidden_states=True,
                return_dict=True,
            )

        decoder_hidden_states = outputs.decoder_hidden_states
        last_hidden = decoder_hidden_states[-1]  # [B, seq_len, hidden_dim]

        audio_name = filename.split("/")[-1].split(".")[0]
        if save_states:
            torch.save(last_hidden.cpu(), f"{self.whisper_rep_path}/{audio_name}_decoder_state.pt")

        transcription = self.processor.batch_decode(
            generated.sequences, skip_special_tokens=True
            )[0]
        return transcription, last_hidden
