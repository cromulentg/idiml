package com.idibon.ml.feature.tokenizer

import com.idibon.ml.feature.{Feature, FeatureTransformer}
import com.idibon.ml.feature.language.LanguageCode
import com.idibon.ml.feature.contenttype.{ContentType, ContentTypeCode}
import com.ibm.icu.util.ULocale

/** Tokenization FeatureTransformer */
class TokenTransformer extends FeatureTransformer {

  /** Tokenizes a String, optionally using language-specific rules
    *
    * If more than one input string is provided, the result will be the
    * concatenation of the tokenized results of all input strings.
    *
    * @param content the string to tokenize, represented as a Feature
    * @param language the primary language for the document
    * @param contentType detected document content type
    * @return all of the tokens in content
    */
  def apply(content: Feature[String], language: Feature[LanguageCode],
    contentType: Feature[ContentType]): Seq[Token] = {
    ICUTokenizer.tokenize(content.get,
      language.get.icuLocale.getOrElse(ULocale.US))
  }
}
