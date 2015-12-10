import org.json4s._
import com.idibon.ml.alloy.Alloy

package com.idibon.ml.feature {

  /** Archivable objects support persistent storage within an Alloy */
  trait Archivable {

    /** Serializes the object within the Alloy
      *
      * Implementations are responsible for persisting any internal state
      * necessary to re-load the object (for example, feature-to-vector
      * index mappings) to the provided Alloy.Writer.
      * 
      * Implementations may return a JsonObject of configuration data
      * to include when re-loading the object.
      */
    def save(writer: Alloy.Writer): Option[JObject]

    /** Reloads the object from the Alloy */
    def load(reader: Alloy.Reader, config: Option[JObject]): this.type
  }
}
