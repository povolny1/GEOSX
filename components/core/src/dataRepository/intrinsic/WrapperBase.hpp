/*
 * DataObjectBase.hpp
 *
 *  Created on: Jun 17, 2016
 *      Author: rrsettgast
 */

#ifndef COMPONENTS_CORE_SRC_DATAREPOSITORY_DATAOBJECTBASE_HPP_
#define COMPONENTS_CORE_SRC_DATAREPOSITORY_DATAOBJECTBASE_HPP_

#include <string>
#include <memory>

namespace asctoolkit
{
namespace sidre
{
class DataView;
}
}

namespace geosx
{
namespace dataRepository
{

class WrapperCollection;

class WrapperBase
{
public:

  /*!
   * \brief default destuctor
   */
  virtual ~WrapperBase();

  /*!
   *
   * @param name name of the object
   * \brief constructor
   */
  explicit WrapperBase( std::string const & name,
                        WrapperCollection * const parent );


  WrapperBase( WrapperBase&& source );


  /*!
   *
   * @return type_info of the DataObject
   */
  virtual std::type_info const & get_typeid() const = 0;


  virtual bool empty() const = 0;
  virtual std::size_t size( ) const = 0;
  virtual void reserve( std::size_t new_cap ) = 0;
  virtual std::size_t capacity() const = 0;
  virtual std::size_t max_size() const = 0;
  virtual void clear() = 0;
  virtual void insert() = 0;
  /*
     iterator erase( iterator pos );
     iterator erase( const_iterator pos );
     iterator erase( const_iterator first, const_iterator last );
     size_type erase( const key_type& key );

     iterator erase( const_iterator pos );
     iterator erase( iterator first, iterator last );
     iterator erase( const_iterator first, const_iterator last );

     void swap( unordered_map& other );
     void swap( vector& other );
   */

  virtual void resize( std::size_t newsize ) = 0;


  int sizedFromParent() const
  {
    return m_sizedFromParent;
  }

  void setSizedFromParent( int val )
  {
    m_sizedFromParent = val;
  }

  asctoolkit::sidre::DataView const * getSidreView() const
  {
    return m_sidreView;
  }
  asctoolkit::sidre::DataView * getSidreView()
  {
    return m_sidreView;
  }

private:
  std::string m_name;
  WrapperCollection* m_parent;
  int m_sizedFromParent;
  asctoolkit::sidre::DataView* m_sidreView;

  WrapperBase() = delete;
  WrapperBase( WrapperBase const & ) = delete;
  WrapperBase& operator=( WrapperBase const & ) = delete;
  WrapperBase& operator=( WrapperBase&& ) = delete;

};

}
} /* namespace geosx */

#endif /* COMPONENTS_CORE_SRC_DATAREPOSITORY_DATAOBJECTBASE_HPP_ */
