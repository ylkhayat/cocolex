# CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation

This is the **official implementation** of **CoCoLex**, a decoding strategy designed to improve faithfulness in legal text generation by dynamically interpolating model-generated token distributions with a copy-based distribution derived from retrieved context, guided by model confidence.

## 📄 Paper

For more details, please refer to our ACL 2025 paper:  
**[CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation](https://aclanthology.org/2025.acl-long.931/)**  
Published in *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)*.

## 📚 Citation

If you use this work, please cite:

```bibtex
@inproceedings{t-y-s-s-etal-2025-cocolex,
    title = "{C}o{C}o{L}ex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation",
    author = "T.y.s.s, Santosh  and
      Elkhayat, Youssef Tarek  and
      Ichim, Oana  and
      Shetty, Pranav  and
      Wang, Dongsheng  and
      Ma, Zhiqiang  and
      Nourbakhsh, Armineh  and
      Liu, Xiaomo",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.931/",
    pages = "19002--19018",
    ISBN = "979-8-89176-251-0",
    abstract = "Due to their ability to process long and complex contexts, LLMs can offer key benefits to the Legal domain, but their adoption has been hindered by their tendency to generate unfaithful, ungrounded, or hallucinatory outputs. While Retrieval-Augmented Generation offers a promising solution by grounding generations in external knowledge, it offers no guarantee that the provided context will be effectively integrated. To address this, context-aware decoding strategies have been proposed to amplify the influence of relevant context, but they usually do not explicitly enforce faithfulness to the context. In this work, we introduce Confidence-guided Copy-based Decoding for Legal Text Generation (CoCoLex){---}a decoding strategy that dynamically interpolates the model produced vocabulary distribution with a distribution derived based on copying from the context. CoCoLex encourages direct copying based on models' confidence, ensuring greater fidelity to the source. Experimental results on five legal benchmarks demonstrate that CoCoLex outperforms existing context-aware decoding methods, particularly in long-form generation tasks."
}
```