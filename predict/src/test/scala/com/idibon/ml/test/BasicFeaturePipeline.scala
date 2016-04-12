package com.idibon.ml.test

import com.idibon.ml.feature._

/** Creates basic feature pipelines for unit tests */
object BasicFeaturePipeline {

  /** Builds and returns an unprimed unigram feature pipeline */
  def classification: FeaturePipeline = {
    (FeaturePipelineBuilder.named("basic")
      += FeaturePipelineBuilder.entry("idx", new indexer.IndexTransformer, "words")
      += FeaturePipelineBuilder.entry("words",
        new bagofwords.BagOfWordsTransformer(Seq(tokenizer.Tag.Word),
          bagofwords.CaseTransform.ToLower), "tokens", "lang")
      += FeaturePipelineBuilder.entry("tokens", new tokenizer.TokenTransformer,
        "content", "lang", "content-type")
      += FeaturePipelineBuilder.entry("lang", new language.LanguageDetector,
        "$document", "content-type")
      += FeaturePipelineBuilder.entry("content-type",
        new contenttype.ContentTypeDetector(false), "$document")
      += FeaturePipelineBuilder.entry("content", new ContentExtractor, "$document")
      := ("idx"))
    }
}
