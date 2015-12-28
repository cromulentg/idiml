import java.io.{DataInputStream, DataOutputStream}
import org.apache.spark.mllib.linalg.Vector

import com.ibm.icu.lang.UCharacter
import com.ibm.icu.lang.UCharacterEnums.ECharacterCategory._

import com.idibon.ml.alloy.Codec
import com.idibon.ml.feature.Feature

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

  case class Token(var content: String, var tag: Tag.Value,
    var offset: Int, var length: Int) extends Feature[Token] {

    // Default parameterless constructor for reflection or when you
    // just want to load a saved Token
    def this() = this(content = "", tag = Tag.Word, offset = 0, length = 0)

    def get = this

    def save(output: DataOutputStream) {
      Codec.String.write(output, content)
      Codec.VLuint.write(output, tag.id)
      Codec.VLuint.write(output, offset)
      Codec.VLuint.write(output, length)
    }

    def load(input: DataInputStream) {
      content = Codec.String.read(input)
      tag = Tag.apply(Codec.VLuint.read(input))
      offset = Codec.VLuint.read(input)
      length = Codec.VLuint.read(input)
    }
  }
}
