package com.idibon.ml.feature.tokenizer

import com.idibon.ml.feature.{Feature, FeatureTransformer}
import com.idibon.ml.feature.language.LanguageCode
import com.ibm.icu.util.ULocale

/** Tokenization FeatureTransformer */
class TokenTransformer extends FeatureTransformer {

  /** Tokenizes a String, optionally using language-specific rules
    *
    * If more than one input string is provided, the result will be the
    * concatenation of the tokenized results of all input strings.
    *
    * @param content - the string to tokenize, represented as a Feature
    * @param language - the primary language for the document
    * @return all of the tokens in content
    */
  def apply(content: Feature[String], language: LanguageCode): Seq[Token] = {
    ICUTokenizer.tokenize(content.get,
      language.icuLocale.getOrElse(ULocale.US))
  }
}
