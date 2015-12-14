import org.json4s.JObject
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
      * Implementations may return a JObject of configuration data
      * to include when re-loading the object.
      *
      * @param writer destination within Alloy for any resources that
      *   must be preserved for this object to be reloadable
      * @return Some[JObject] of configuration data that must be preserved
      *   to reload the object. None if no configuration is needed
      */
    def save(writer: Alloy.Writer): Option[JObject]

    /** Reloads the object from the Alloy
      *
      * @param reader location within Alloy for loading any resources
      *   previous preserved by a call to
      *   {@link com.idibon.ml.feature.Archivable#save}
      * @param config archived configuration data returned by a previous
      *   call to {@link com.idibon.ml.feature.Archivable#save}
      * @return this object
      */
    def load(reader: Alloy.Reader, config: Option[JObject]): this.type
  }
}
