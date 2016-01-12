package com.idibon.ml.feature

import org.json4s.JObject
import com.idibon.ml.alloy.Alloy
import com.idibon.ml.common.Reflect

/** Archivable objects support persistent storage within an Alloy
  *
  * Implementations must be paired with an ArchiveLoader implementation that
  * load a previously-persisted instance from the Alloy.
  */
trait Archivable[T <: Archivable[T, ArchiveLoader[T]], +U <: ArchiveLoader[T]] {

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
}

trait ArchiveLoader[T] {
  /** Reloads the object from the Alloy
    *
    * @param reader location within Alloy for loading any resources
    *   previous preserved by a call to
    *   {@link com.idibon.ml.feature.Archivable#save}
    * @param config archived configuration data returned by a previous
    *   call to {@link com.idibon.ml.feature.Archivable#save}
    * @return this object
    */
  def load(reader: Alloy.Reader, config: Option[JObject]): T
}

object Archivable {
  /** Saves an object, if the object is Archivable
    *
    * Checks if the provided object is Archivable, and if so archives
    * it to the provided writer using the {@link Archivable#save}
    * method on the object, returning the configuration JObject from
    * the call to save. If the object is not Archivable, returns None.
    *
    * @param obj - an object which may need archiving
    * @param writer - an alloy writer configured to store
    *    resources for obj
    */
  def save[T](obj: T, writer: Alloy.Writer): Option[JObject] = {
    obj match {
      case archive: Archivable[_, _] => archive.save(writer)
      case _ => None
    }
  }
}

object ArchiveLoader {

  /** Loads an instance of an Archivable class using its paired ArchiveLoader
    *
    * If class is not Archivable, returns None
    *
    * @param class - The Class of the Archivable object to reify
    * @param reader - A reader configured to load the resources for the
    *    loaded object
    * @param config - Any configuration meta-data for the object.
    */
  def reify[T](`class`: Class[_], reader: Alloy.Reader,
    config: Option[JObject]): Option[T] = {

    /* check if the class actually implements Archivable; if so, extract the
     * type arguments used to specialize Archivable to get the paired loader
     */
    Reflect.getTypeParametersAs[Archivable[_, _]](`class`) match {
      case Nil => None
      case args => {
        /* the last entry in the args list is the loader class. instantiate
         * a builder and call the load method to load the object */
        Some(args.last.newInstance.asInstanceOf[ArchiveLoader[_]]
          .load(reader, config).asInstanceOf[T])
      }
    }
  }
}
