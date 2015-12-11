import java.io.{DataInputStream, DataOutputStream}
import org.apache.spark.mllib.linalg.Vector

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
  }

  case class Token(var content: String, var tag: Tag.Value,
    var offset: Int, var length: Int) extends Feature[Token] {

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
