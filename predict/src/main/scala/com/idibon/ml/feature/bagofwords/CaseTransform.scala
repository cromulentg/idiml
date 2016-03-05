package com.idibon.ml.feature.bagofwords

import scala.collection.generic.{CanBuildFrom, GenericCompanion}

import com.idibon.ml.feature.Feature
import com.idibon.ml.feature.language.LanguageCode
import com.idibon.ml.feature.tokenizer.{Token, Tag}
import com.ibm.icu.lang.UCharacter

/** Provides case-transformation capabilities to bag-of-words classes */
trait CaseTransform {
  val transform: CaseTransform.Value

  /** Returns a language-specific transformation function
    *
    * Given a target language and the desired case transformation, returns
    * a function that performs case transformation to convert tokens to
    * words
    *
    * @param lang the document's detected language
    * @return a function to transform tokens into words
    */
  def transformation(lang: Feature[LanguageCode]): Function1[Token, Word] = {
    transform match {
      case CaseTransform.ToLower => {
        lang.get.icuLocale.map(locale => {
          (t: Token) => Word(UCharacter.toLowerCase(locale, t.content))
        }).getOrElse({
          (t: Token) => Word(UCharacter.toLowerCase(t.content))
        })
      }
      case CaseTransform.ToUpper => {
        lang.get.icuLocale.map(locale => {
          (t: Token) => Word(UCharacter.toUpperCase(locale, t.content))
        }).getOrElse({
          (t: Token) => Word(UCharacter.toLowerCase(t.content))
        })
      }
      case CaseTransform.None => {
        (t: Token) => Word(t.content)
      }
    }
  }
}

object CaseTransform extends Enumeration {
  // TODO: add more case fold choices here
  val ToLower,  /* convert all words to lower case */
    ToUpper,    /* convert all words to upper case */
    None        /* no change in case */ = Value
}
