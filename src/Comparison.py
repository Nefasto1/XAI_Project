import torch as th
import numpy as np

from lxt.models.llama import attnlrp
import lxt.functional as lf

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from pathlib import Path
import os

import inseq
import shap
import subprocess

class Comparison():
    def __init__(self, model_attn, model_integrated, tokenizer):
        self.tokenizer  = tokenizer

        self.model_attn       = model_attn        
        self.model_integrated = model_integrated

    def _get_output(self, prompt):
        input_ids  = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(self.model_attn.device)
        output_ids = self.model_attn.generate(input_ids, max_length=200, num_return_sequences=1)
        tokens     = self.tokenizer.convert_ids_to_tokens(output_ids[0])
        string     = self.tokenizer.convert_tokens_to_string(tokens)

        return string
        
    def _AttnLRP(self, prompt):
        model = self.model_attn

        input_ids    = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
        input_embeds = model.get_input_embeddings()(input_ids)
        input_embeds = input_embeds.requires_grad_()
        input_embeds.retain_grad()
        
        attnlrp.register(model)

        prova = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False)
        
        output_logits           = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
        max_logits, max_indices = th.max(output_logits[0, -1, :], dim=-1)
        
        max_logits.backward(max_logits)
        relevance = input_embeds.grad.float().sum(-1).cpu()[0]
        prova = relevance
        
        # remove '_' characters from token strings
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        tokens = np.array(tokens)#[idxs_to_keep]
        
        # Drop the "▁" token
        relevance = relevance / relevance.abs().max()
        
        return tokens, relevance.numpy()#, prova

    def _get_tokens(self, output):
        tokens = [token.token for token in output.sequence_attributions[0].source]

        return tokens
        
    def _Shap(self, prompt):
        model = self.model_integrated
        tokenizer = self.tokenizer
        
        explainer = shap.Explainer(model, tokenizer)
        shap_values = explainer([prompt])

        relevance = shap_values.values[0, :, 0]
        relevance = th.from_numpy(relevance)
        relevance /= relevance.abs().max() 

        return relevance.numpy()

    def _get_attributions(self, prompt, generation_args, attribution_type):
        model  = inseq.load_model("gpt2", attribution_type)
        out    = model.attribute(input_texts=prompt, generation_args=generation_args)

        tokens = self._get_tokens(out)
        
        relevance = out.aggregate("mean").sequence_attributions[0].target_attributions[:, 0]
        relevance = relevance[ (1 - relevance.isnan().int()).bool()]
        relevance /= relevance.abs().max() 
        
        return np.array(tokens), relevance.numpy()#, out

    def __call__(self, prompt, generation_args):        
        # AttnLRP
        Attn_tokens, Attn_relevance             = self._AttnLRP(prompt)
        
        # Shap
        Shap_tokens, Shap_relevance             = Attn_tokens, self._Shap(prompt)
        
        # Integrated Gradients
        Integrated_tokens, Integrated_relevance = self._get_attributions(prompt, generation_args, "integrated_gradients")

        # Gradient X Input
        Gradient_tokens, Gradient_relevance     = self._get_attributions(prompt, generation_args, "input_x_gradient")
        
        # DeepLift
        DeepLift_tokens, DeepLift_relevance     = self._get_attributions(prompt, generation_args, "deeplift")
        
        # Lime
        Lime_tokens, Lime_relevance             = self._get_attributions(prompt, generation_args, "lime")

        # Gradient_Shap
        Gradient_Shap_tokens, Gradient_Shap_relevance = self._get_attributions(prompt, generation_args, "gradient_shap")

        output = self._get_output(prompt)
        
        return {"AttnLRP":              [Attn_tokens,          Attn_relevance], 
                "Integrated_Gradients": [Integrated_tokens,    Integrated_relevance],
                "Gradient_X_Input":     [Gradient_tokens,      Gradient_relevance],
                "DeepLift":             [DeepLift_tokens,      DeepLift_relevance],
                "Gradient_Shap":        [Gradient_Shap_tokens, Gradient_Shap_relevance],
                "Lime":                 [Lime_tokens,          Lime_relevance],
                "Shap":                 [Shap_tokens,          Shap_relevance]
               }, output 

    def plot(self, dictionary):
        for key in dictionary.keys():
            print(key)
            relevance = dictionary[key][1]
            names     = [name.replace("Ġ", "").replace("Ċ", "\n") + f"{i}" for i, name in enumerate(dictionary[key][0])]
            colors    = ["green" if value > 0 else "red" for value in relevance]
            
            plt.figure(figsize=(10, 10))
            plt.xticks(rotation=90)
            plt.bar(names, relevance, color=colors)
            plt.title(key)
            plt.show()
    
    def _apply_colormap(self, relevance, cmap="bwr"):
        colormap = cm.get_cmap(cmap)
        return colormap(colors.Normalize(vmin=-1, vmax=1)(relevance))

    def _to_latex(self, words, relevances, name):
        # Generate LaTeX code
        latex_code = r'''
        \documentclass[arwidth=200mm]{standalone} 
        \usepackage[dvipsnames]{xcolor}
        
        \begin{document}
        \fbox{
        \parbox{\textwidth}{
        \setlength\fboxsep{0pt}
        '''
    
        for word, relevance in zip(words, relevances):
            rgb = self._apply_colormap(relevance)
            r, g, b = int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)

            if word == "Ċ":
                latex_code += "\\\\"
            else:
                word = word.replace("Ġ", "").replace("$", "\\$").replace("&", "\\&").replace("%", "\\%")
                latex_code += f' \\colorbox[RGB]{{{r},{g},{b}}}{{\\strut {word}}}'
    
    
        latex_code += r'}}\end{document}'
    
        # Save LaTeX code to a file
        path = Path(f"pdf/{name}.tex")
        os.makedirs(path.parent, exist_ok=True)
    
        with open(path.with_suffix(".tex"), 'w') as f:
            f.write(latex_code)
            
    def to_latex(self, dictionary, name=""):
        for key in dictionary.keys():
            self._to_latex(dictionary[key][0], dictionary[key][1], name+key)

    def to_pdf(self, dictionary, name=""):
        self.to_latex(dictionary, name)
        subprocess.Popen("./compile.sh", shell=True, stdout=subprocess.DEVNULL)