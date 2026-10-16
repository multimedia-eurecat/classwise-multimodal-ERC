"""Light-ASD wrapper using the vendored source network, without sys.path edits."""
import torch

from unimodal.visual.asd import LightASDWrapper as _Wrapper


class LightASDWrapper(_Wrapper):
    def _load_model(self, weights_path, device):
        from .third_party.light_asd.Model import ASD_Model
        from .third_party.light_asd.loss import lossAV

        backbone = ASD_Model().to(device)
        classifier = lossAV().to(device)
        state = torch.load(weights_path, map_location=device, weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        backbone.load_state_dict({k[len("model."):]:v for k,v in state.items() if k.startswith("model.")})
        classifier.load_state_dict({k[len("lossAV."):]:v for k,v in state.items() if k.startswith("lossAV.")})
        backbone.eval()
        classifier.eval()
        return {"backbone":backbone, "classifier":classifier}
