package com.idibon.ml.feature

import java.io._
import scala.collection.mutable.HashMap
import com.idibon.ml.alloy.Codec
import com.idibon.ml.common.Reflect;

/** Adds methods to save Feature instances to OutputStreams
  *
  * Features are loaded and saved using reflection. To conserve space, rather
  * than recording the class name for every feature, feature class names are
  * indexed using an internal history table. Up to 128 previously-observed
  * features are stored in a local cache. When a feature is stored, if a
  * cached entry already exists for the feature's class name, only the
  * index is stored, rather than the entire class name. When a new feature
  * is encountered, the entire class name is saved and the feature is
  * added to the cache for future features.
  *
  * This encoding is performed with a 1-byte value prepended to the feature
  * data; negative values indicate "use the cached class name at cache index
  * (128 + value).
  *
  * @param out - Underlying output stream where features are written
  * @param maxCacheSize - controls the maximum cache size, must be between
  *    1 and 128. primarily used by unit tests
  */
class FeatureOutputStream(out: OutputStream, maxCacheSize: Int = 128)
    extends DataOutputStream(out) {

  private[this] val _classNameCache = HashMap[String, Int]()

  /** Writes a Feature to the OutputStream.
    *
    * @param f - Feature to write
    */
  def writeFeature(f: Feature[_] with Buildable[_, _]) {
    val className = f.getClass.getName
    _classNameCache.get(className) match {
      case Some(index) => {
        writeByte(index - 128)
      }
      case None => {
        /* FIXME: this is an uber-cheesy replacement strategy -- if the
         * cache gets too large, all entries are evicted. it probably
         * won't matter too much in practice, since 129 feature types
         * seems a bit excessive */
        if (_classNameCache.size == maxCacheSize)
          _classNameCache.clear
        val index = _classNameCache.size
        writeByte(index)
        Codec.String.write(this, className)
        _classNameCache += (className -> index)
      }
    }
    f.save(this)
  }
}

/** Adds ability to load Feature instances from an InputStream
  *
  * Features must have been previously saved using FeatureOutputStream
  */
class FeatureInputStream(in: InputStream) extends DataInputStream(in) {

  private[this] val _classNameCache = HashMap[Int, Builder[Feature[_]]]()

  /** Reads a Feature from the InputStream
    *
    * @return - the loaded feature
    */
  def readFeature: Feature[_] = {
    val builder = readByte match {
      case cached if cached < 0 => _classNameCache(128 + cached)
      case uncached => {
        val args = Reflect.getTypeParametersAs[Buildable[_, _]](
          Class.forName(Codec.String.read(this)))

        _classNameCache += (uncached.toInt ->
          args.last.newInstance.asInstanceOf[Builder[Feature[_]]])

        _classNameCache(uncached)
      }
    }
    builder.build(this)
  }
}
