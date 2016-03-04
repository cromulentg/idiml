package com.idibon.ml.feature.bagofwords

import com.idibon.ml.feature.{Feature, FeatureTransformer, Chain}
import com.idibon.ml.feature.tokenizer.{Token, Tag}
import com.idibon.ml.feature.language.LanguageCode
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.alloy.Alloy

import org.json4s.{JObject, JString}
import org.json4s.JsonDSL._

/** BagOfWords-style transformation for sequence classifiers
  *
  * Produces a chain of word features from a chain of tokens, optionally
  * applying a language-specific case transformation operation. This
  * transform is structure-preserving, so may be used in the feature
  * extraction graph of sequence classifiers.
  *
  * @param transform case folding operation
  */
class ChainBagOfWords(val transform: CaseTransform.Value)
    extends FeatureTransformer with CaseTransform
    with Archivable[ChainBagOfWords, ChainBagOfWordsLoader] {

  def apply(c: Chain[Feature[Token]], lc: Feature[LanguageCode]): Chain[Word] = {
    val f = transformation(lc)
    c.map(t => f(t.value.get))
  }

  def save(writer: Alloy.Writer): Option[JObject] =
    Some(("transform" -> transform.toString))
}

/** Paired loader class for ChainBagOfWords */
class ChainBagOfWordsLoader extends ArchiveLoader[ChainBagOfWords] {
  /** Loads the transform from the alloy and config state */
  def load(e: Engine, r: Option[Alloy.Reader], config: Option[JObject]) = {
    val transform = (config.get \ "transform").asInstanceOf[JString].s
    new ChainBagOfWords(CaseTransform.withName(transform))
  }
}
