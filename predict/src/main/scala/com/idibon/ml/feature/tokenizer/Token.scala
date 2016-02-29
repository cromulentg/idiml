import com.ibm.icu.lang.UCharacter
import com.ibm.icu.lang.UCharacterEnums.ECharacterCategory._

import com.idibon.ml.alloy.Codec
import com.idibon.ml.feature._

package com.idibon.ml.feature.tokenizer {

  /** Used to specify the types of tokens that may be generated
    *
    * Every token is assigned a type during tokenization (whitespace,
    * word, punctuation, etc.).
    *
    * NB: Tokens serialize tags as integer indices, so only add new values
    * to the end of the list!
    */
  object Tag extends Enumeration {
    val Word,
      Punctuation,
      Whitespace = Value

    /** Return the Tag (type) of the provided token content.
      *
      * @param content the content within the token
      * @return the Tag of the token
      */
    def of(content: String) = {
      if (content.forall(UCharacter.isUWhiteSpace(_)))
        Whitespace
      else if (content.forall(isPunctuation(_)))
        Punctuation
      else
        Word
    }

    /** Returns true if the character is a punctuation mark
      *
      * The list of unicode character categories that are mapped onto the
      * "punctuation" test is taken from
      *  http://icu-project.org/apiref/icu4j/com/ibm/icu/lang/UCharacter.html
      */
    private [tokenizer] def isPunctuation(ch: Char) = {
      UCharacter.getType(ch) match {
        case DASH_PUNCTUATION
           | START_PUNCTUATION
           | END_PUNCTUATION
           | CONNECTOR_PUNCTUATION
           | OTHER_PUNCTUATION
           | INITIAL_PUNCTUATION
           | FINAL_PUNCTUATION => true
        case _ => false
      }
    }
  }

  case class Token(content: String, tag: Tag.Value, offset: Int, length: Int)
      extends Feature[Token] with Buildable[Token, TokenBuilder]{

    def get = this

    def save(output: FeatureOutputStream) {
      Codec.String.write(output, content)
      Codec.VLuint.write(output, tag.id)
      Codec.VLuint.write(output, offset)
      Codec.VLuint.write(output, length)
    }

    def getHumanReadableString: Option[String] = {
      Some(s"${this.content} (${this.offset}, ${this.length})")
    }
  }

  class TokenBuilder extends Builder[Token] {
    def build(input: FeatureInputStream) = {
      new Token(Codec.String.read(input),
        Tag.apply(Codec.VLuint.read(input)),
        Codec.VLuint.read(input),
        Codec.VLuint.read(input))
    }
  }
}
