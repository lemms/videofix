# - Try to find OpenCV library installation
# See http://sourceforge.net/projects/opencvlibrary/
#
# The follwoing variables are optionally searched for defaults
#  OpenCV_ROOT_DIR:            Base directory of OpenCv tree to use.
#  OpenCV_FIND_REQUIRED_COMPONENTS : FIND_PACKAGE(OpenCV COMPONENTS ..) 
#    compatible interface. typically  CV CXCORE CVAUX HIGHGUI CVCAM .. etc.
#
# The following are set after configuration is done: 
#  OpenCV_FOUND
#  OpenCV_INCLUDE_DIR
#  OpenCV_LIBRARIES
#  OpenCV_LINK_DIRECTORIES
#
# 2004/05 Jan Woetzel, Friso, Daniel Grest 
# 2006/01 complete rewrite by Jan Woetzel
# 1006/09 2nd rewrite introducing ROOT_DIR and PATH_SUFFIXES 
#   to handle multiple installed versions gracefully by Jan Woetzel
#
# tested with:
# -OpenCV 0.97 (beta5a):  MSVS 7.1, gcc 3.3, gcc 4.1
# -OpenCV 0.99 (1.0rc1):  MSVS 7.1
#
# www.mip.informatik.uni-kiel.de/~jw
#
# ---------------------
#
# $Id: FindOpenCV.cmake 8 2009-01-04 21:13:48Z adeguet1 $
# ERC-CISST version:
#  
# - Removed deprecated code starting with cap. OPENCV
# - Removed debugging code and messages
# - Removed path and options specifics to previous authors setups
#
# This file should be removed when CMake will provide an equivalent

message("Searching for OpenCV ${OpenCV_FIND_VERSION}")


# required cv components with header and library if COMPONENTS unspecified
IF(NOT OpenCV_FIND_COMPONENTS)
  # default



  #
  # fix for OpenCV 2.4.6.1 on Mac OSX
  #
  #SET(OpenCV_FIND_REQUIRED_COMPONENTS CV CXCORE HIGHGUI)
  SET(OpenCV_FIND_REQUIRED_COMPONENTS CORE HIGHGUI VIDEO VIDEOIO VIDEOSTAB IMGCODECS IMGPROC PHOTO)



#  IF(WIN32)
#    LIST(APPEND OpenCV_FIND_REQUIRED_COMPONENTS  CVCAM ) # WIN32 only actually
#  ENDIF(WIN32)  
ENDIF(NOT OpenCV_FIND_COMPONENTS)


# typical root dirs of installations, exactly one of them is used
SET(OpenCV_POSSIBLE_ROOT_DIRS
  "${OpenCV_ROOT_DIR}"
  "$ENV{OpenCV_ROOT_DIR}"  
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Intel(R) Open Source Computer Vision Library_is1;Inno Setup: App Path]"
  "$ENV{ProgramFiles}/OpenCV"
  "/usr/local"
  "/usr"
	"/opt/local"
  )
  
#
# select exactly ONE OpenCV base directory/tree 
# to avoid mixing different version headers and libs
#
FIND_PATH(OpenCV_ROOT_DIR 
  NAMES 
  cv/include/cv.h     # windows
  include/opencv/cv.h # linux /opt/net
  include/cv/cv.h 
  include/cv.h 
  PATHS ${OpenCV_POSSIBLE_ROOT_DIRS})

MESSAGE("OpenCV_ROOT_DIR: ${OpenCV_ROOT_DIR}")

# header include dir suffixes appended to OpenCV_ROOT_DIR
SET(OpenCV_INCDIR_SUFFIXES
  include
  include/cv
  include/opencv
  include/opencv2
  cv/include
  cxcore/include
  cvaux/include
  otherlibs/cvcam
  otherlibs/highgui
  )

# library linkdir suffixes appended to OpenCV_ROOT_DIR 
SET(OpenCV_LIBDIR_SUFFIXES
  lib
  OpenCV/lib
  x86/vc14/lib
  )

# find incdir for each lib
FIND_PATH(OpenCV_CV_INCLUDE_DIR
  NAMES cv.h      
  PATHS ${OpenCV_ROOT_DIR} 
  PATH_SUFFIXES ${OpenCV_INCDIR_SUFFIXES})

FIND_PATH(OpenCV_CXCORE_INCLUDE_DIR   
  NAMES cxcore.h
  PATHS ${OpenCV_ROOT_DIR} 
  PATH_SUFFIXES ${OpenCV_INCDIR_SUFFIXES})
FIND_PATH(OpenCV_CVAUX_INCLUDE_DIR    
  NAMES cvaux.h
  PATHS ${OpenCV_ROOT_DIR} 
  PATH_SUFFIXES ${OpenCV_INCDIR_SUFFIXES})
FIND_PATH(OpenCV_HIGHGUI_INCLUDE_DIR  
  NAMES highgui.h 
  PATHS ${OpenCV_ROOT_DIR} 
  PATH_SUFFIXES ${OpenCV_INCDIR_SUFFIXES})
FIND_PATH(OpenCV_CVCAM_INCLUDE_DIR    
  NAMES cvcam.h 
  PATHS ${OpenCV_ROOT_DIR} 
  PATH_SUFFIXES ${OpenCV_INCDIR_SUFFIXES})



#
# fix for OpenCV 2.4.6.1 on Mac OSX
#
FIND_PATH(OpenCV_CORE_INCLUDE_DIR   
  NAMES opencv2/opencv.hpp
  PATHS ${OpenCV_ROOT_DIR} 
  PATH_SUFFIXES ${OpenCV_INCDIR_SUFFIXES})

FIND_PATH(OpenCV_VIDEO_INCLUDE_DIR   
  NAMES opencv2/video.hpp
  PATHS ${OpenCV_ROOT_DIR} 
  PATH_SUFFIXES ${OpenCV_INCDIR_SUFFIXES})

FIND_PATH(OpenCV_VIDEOIO_INCLUDE_DIR   
  NAMES opencv2/videoio.hpp
  PATHS ${OpenCV_ROOT_DIR} 
  PATH_SUFFIXES ${OpenCV_INCDIR_SUFFIXES})

FIND_PATH(OpenCV_VIDEOSTAB_INCLUDE_DIR   
  NAMES opencv2/videostab.hpp
  PATHS ${OpenCV_ROOT_DIR} 
  PATH_SUFFIXES ${OpenCV_INCDIR_SUFFIXES})

FIND_PATH(OpenCV_IMGCODECS_INCLUDE_DIR   
  NAMES opencv2/imgcodecs.hpp
  PATHS ${OpenCV_ROOT_DIR} 
  PATH_SUFFIXES ${OpenCV_INCDIR_SUFFIXES})

FIND_PATH(OpenCV_IMGPROC_INCLUDE_DIR   
  NAMES opencv2/imgproc.hpp
  PATHS ${OpenCV_ROOT_DIR} 
  PATH_SUFFIXES ${OpenCV_INCDIR_SUFFIXES})

FIND_PATH(OpenCV_PHOTO_INCLUDE_DIR   
  NAMES opencv2/photo.hpp
  PATHS ${OpenCV_ROOT_DIR} 
  PATH_SUFFIXES ${OpenCV_INCDIR_SUFFIXES})

#
# find sbsolute path to all libraries 
# some are optionally, some may not exist on Linux
#
FIND_LIBRARY(OpenCV_CV_LIBRARY   
  NAMES cv opencv cv${OpenCV_FIND_VERSION} opencv${OpenCV_FIND_VERSION}
  PATHS ${OpenCV_ROOT_DIR}  
  PATH_SUFFIXES ${OpenCV_LIBDIR_SUFFIXES})
FIND_LIBRARY(OpenCV_CVAUX_LIBRARY
  NAMES cvaux cvaux${OpenCV_FIND_VERSION}
  PATHS ${OpenCV_ROOT_DIR}
  PATH_SUFFIXES ${OpenCV_LIBDIR_SUFFIXES})
FIND_LIBRARY(OpenCV_CVCAM_LIBRARY   
  NAMES cvcam cvcam${OpenCV_FIND_VERSION}
  PATHS ${OpenCV_ROOT_DIR}
  PATH_SUFFIXES ${OpenCV_LIBDIR_SUFFIXES}) 
FIND_LIBRARY(OpenCV_CVHAARTRAINING_LIBRARY
  NAMES cvhaartraining cvhaartraining${OpenCV_FIND_VERSION}
  PATHS ${OpenCV_ROOT_DIR}
  PATH_SUFFIXES ${OpenCV_LIBDIR_SUFFIXES}) 
FIND_LIBRARY(OpenCV_CXCORE_LIBRARY  
  NAMES cxcore cxcore${OpenCV_FIND_VERSION}
  PATHS ${OpenCV_ROOT_DIR}  PATH_SUFFIXES ${OpenCV_LIBDIR_SUFFIXES})
FIND_LIBRARY(OpenCV_CXTS_LIBRARY   
  NAMES cxts cxts${OpenCV_FIND_VERSION}
  PATHS ${OpenCV_ROOT_DIR}
  PATH_SUFFIXES ${OpenCV_LIBDIR_SUFFIXES})
FIND_LIBRARY(OpenCV_HIGHGUI_LIBRARY  
  NAMES highgui highgui${OpenCV_FIND_VERSION}
  PATHS ${OpenCV_ROOT_DIR}
  PATH_SUFFIXES ${OpenCV_LIBDIR_SUFFIXES})
FIND_LIBRARY(OpenCV_TRS_LIBRARY  
  NAMES trs trs${OpenCV_FIND_VERSION}
  PATHS ${OpenCV_ROOT_DIR}  PATH_SUFFIXES ${OpenCV_LIBDIR_SUFFIXES})



FIND_LIBRARY(OpenCV_CORE_LIBRARY  
  NAMES opencv_core opencv_core${OpenCV_FIND_VERSION}
  PATHS ${OpenCV_ROOT_DIR}  PATH_SUFFIXES ${OpenCV_LIBDIR_SUFFIXES})

FIND_LIBRARY(OpenCV_HIGHGUI_LIBRARY  
  NAMES opencv_highgui opencv_highgui${OpenCV_FIND_VERSION}
  PATHS ${OpenCV_ROOT_DIR}  PATH_SUFFIXES ${OpenCV_LIBDIR_SUFFIXES})

FIND_LIBRARY(OpenCV_VIDEO_LIBRARY  
  NAMES opencv_video opencv_video${OpenCV_FIND_VERSION}
  PATHS ${OpenCV_ROOT_DIR}  PATH_SUFFIXES ${OpenCV_LIBDIR_SUFFIXES})

FIND_LIBRARY(OpenCV_VIDEOIO_LIBRARY  
  NAMES opencv_videoio opencv_videoio${OpenCV_FIND_VERSION}
  PATHS ${OpenCV_ROOT_DIR}  PATH_SUFFIXES ${OpenCV_LIBDIR_SUFFIXES})

FIND_LIBRARY(OpenCV_VIDEOSTAB_LIBRARY  
  NAMES opencv_videostab opencv_videostab${OpenCV_FIND_VERSION}
  PATHS ${OpenCV_ROOT_DIR}  PATH_SUFFIXES ${OpenCV_LIBDIR_SUFFIXES})

FIND_LIBRARY(OpenCV_IMGCODECS_LIBRARY  
  NAMES opencv_imgcodecs opencv_imgcodecs${OpenCV_FIND_VERSION}
  PATHS ${OpenCV_ROOT_DIR}  PATH_SUFFIXES ${OpenCV_LIBDIR_SUFFIXES})

FIND_LIBRARY(OpenCV_IMGPROC_LIBRARY  
  NAMES opencv_imgproc opencv_imgproc${OpenCV_FIND_VERSION}
  PATHS ${OpenCV_ROOT_DIR}  PATH_SUFFIXES ${OpenCV_LIBDIR_SUFFIXES})

FIND_LIBRARY(OpenCV_PHOTO_LIBRARY  
  NAMES opencv_photo opencv_photo${OpenCV_FIND_VERSION}
  PATHS ${OpenCV_ROOT_DIR}  PATH_SUFFIXES ${OpenCV_LIBDIR_SUFFIXES})


# Logic selecting required libs and headers
SET(OpenCV_FOUND ON)
FOREACH(NAME ${OpenCV_FIND_REQUIRED_COMPONENTS})
  # only good if header and library both found   
  IF(NOT OpenCV_${NAME}_INCLUDE_DIR)
    MESSAGE("Unable to find OpenCV_${NAME}_INCLUDE_DIR")
  ENDIF(NOT OpenCV_${NAME}_INCLUDE_DIR)
  IF(NOT OpenCV_${NAME}_LIBRARY)
    MESSAGE("Unable to find OpenCV_${NAME}_LIBRARY")
  ENDIF(NOT OpenCV_${NAME}_LIBRARY)
  IF(OpenCV_${NAME}_INCLUDE_DIR AND OpenCV_${NAME}_LIBRARY)
    LIST(FIND OpenCV_INCLUDE_DIRS "${OpenCV_${NAME}_INCLUDE_DIR}" DIR_FOUND)
    IF(DIR_FOUND EQUAL -1)
        LIST(APPEND OpenCV_INCLUDE_DIRS "${OpenCV_${NAME}_INCLUDE_DIR}")
    ENDIF(DIR_FOUND EQUAL -1)
    LIST(APPEND OpenCV_LIBRARIES    "${OpenCV_${NAME}_LIBRARY}")
  ELSE(OpenCV_${NAME}_INCLUDE_DIR AND OpenCV_${NAME}_LIBRARY)
    SET(OpenCV_FOUND OFF)
  ENDIF(OpenCV_${NAME}_INCLUDE_DIR AND OpenCV_${NAME}_LIBRARY)
ENDFOREACH(NAME)

message("Include dir: ${OpenCV_FOUND}")

# get the link directory for rpath to be used with LINK_DIRECTORIES: 
IF(OpenCV_CV_LIBRARY)
  GET_FILENAME_COMPONENT(OpenCV_LINK_DIRECTORIES ${OpenCV_CV_LIBRARY} PATH)
ENDIF(OpenCV_CV_LIBRARY)

MARK_AS_ADVANCED(
  OpenCV_ROOT_DIR
  OpenCV_INCLUDE_DIRS
  OpenCV_CV_INCLUDE_DIR
  OpenCV_CXCORE_INCLUDE_DIR
  OpenCV_CVAUX_INCLUDE_DIR
  OpenCV_CVCAM_INCLUDE_DIR
  OpenCV_HIGHGUI_INCLUDE_DIR
  OpenCV_LIBRARIES
  OpenCV_CV_LIBRARY
  OpenCV_CXCORE_LIBRARY
  OpenCV_CVAUX_LIBRARY
  OpenCV_CVCAM_LIBRARY
  OpenCV_CVHAARTRAINING_LIBRARY
  OpenCV_CXTS_LIBRARY
  OpenCV_HIGHGUI_LIBRARY
  OpenCV_TRS_LIBRARY
 )


# display help message
IF(NOT OpenCV_FOUND)
  # make FIND_PACKAGE friendly
  IF(OpenCV_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR
            "OpenCV required but some headers or libs not found. Please specify it's location with OpenCV_ROOT_DIR env. variable.")
  ELSE(OpenCV_FIND_REQUIRED)
    MESSAGE(STATUS 
            "ERROR: OpenCV was not found.")
  ENDIF(OpenCV_FIND_REQUIRED)
ELSE(NOT OpenCV_FOUND)
  MESSAGE(STATUS
          "OpenCV found at ${OpenCV_ROOT_DIR}")
ENDIF(NOT OpenCV_FOUND)


#
# $Log: FindOpenCV.cmake,v $
# Revision 1.2  2008/11/03 16:40:34  vagvoba
# cmake: Proper cvcam and highgui directories added to FindOpenCV.cmake
#
# Revision 1.1  2007/07/12 00:22:25  anton
# cmake utilities: Added FindOpenCV.cmake based on example found all over the
# web.  Will have to be removed when CMake provides one.
#
#

