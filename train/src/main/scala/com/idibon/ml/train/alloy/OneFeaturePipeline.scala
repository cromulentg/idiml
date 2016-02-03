package com.idibon.ml.train.alloy

import com.idibon.ml.feature.bagofwords.{CaseTransform, BagOfWordsTransformer}
import com.idibon.ml.feature.indexer.IndexTransformer
import com.idibon.ml.feature.language.LanguageDetector
import com.idibon.ml.feature.ngram.NgramTransformer
import com.idibon.ml.feature.tokenizer.{TokenTransformer, Tag}
import com.idibon.ml.feature.{ContentExtractor, FeaturePipelineBuilder, FeaturePipeline}
import org.json4s.JObject

/**
  * Trait that deals with creating a single feature pipeline from configuration data.
  */
trait OneFeaturePipeline {

  /**
    * Creates a single feature pipeline from the passed in configuration.
    *
    * @param config
    * @return
    */
  def createFeaturePipeline(config: JObject): FeaturePipeline = {
    implicit val formats = org.json4s.DefaultFormats
    val ngramSize = (config \ "ngram").extract[Int]
    //TODO: unhardcode this
    (FeaturePipelineBuilder.named("pipeline")
      += FeaturePipelineBuilder.entry("convertToIndex", new IndexTransformer, "ngrams")
      += FeaturePipelineBuilder.entry("ngrams", new NgramTransformer(1, ngramSize), "bagOfWords")
      += FeaturePipelineBuilder.entry("bagOfWords",
      new BagOfWordsTransformer(List(Tag.Word, Tag.Punctuation), CaseTransform.ToLower),
      "convertToTokens", "languageDetector")
      += FeaturePipelineBuilder.entry("convertToTokens", new TokenTransformer, "contentExtractor", "languageDetector")
      += FeaturePipelineBuilder.entry("languageDetector", new LanguageDetector, "$document")
      += FeaturePipelineBuilder.entry("contentExtractor", new ContentExtractor, "$document")
      := "convertToIndex")
  }
}
