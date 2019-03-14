import os
import platform
import re

from sqlalchemy import create_engine
from sqlalchemy.event import listen
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import create_session, scoped_session, sessionmaker
from sqlalchemy.sql import select, func


# adapted from http://flask.pocoo.org/snippets/22/
_engine = None
_mod_spatialite_path = None
Session = None


def set_sqlite_pragma(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    # store temp files in memory rather then in /tmp.
    # given enough memory, this can avoid "database or disk is full" errors
    # alternative:
    # "PRAGMA temp_store=1"
    # "PRAGMA temp_store_directory='<directory_with_enough_space>'"
    cursor.execute("PRAGMA temp_store=2")
    cursor.close()


def load_spatialite(dbapi_conn, connection_record):
    dbapi_conn.enable_load_extension(True)

    print("INFO: load mod_spatialite from '%s'..." % _mod_spatialite_path)

    if platform.system() == "Windows":
        os.environ['PATH'] = _mod_spatialite_path + ';' + os.environ['PATH']
        dbapi_conn.load_extension('mod_spatialite.dll')
    elif platform.system() == "Linux":
        # we have to load the required libraries manually as
        # os.environ['LD_LIBRARY_PATH'] is not available within the Python
        # environment under Linux
        from ctypes import cdll
        for lib_name in os.listdir(_mod_spatialite_path):
            if re.match(r'.*so(\.\d{1,2})?$', lib_name):
                if lib_name in ['mod_spatialite.so', 'libspatialite.so']:
                    continue
                cdll.LoadLibrary(os.path.join(_mod_spatialite_path, lib_name))

        dbapi_conn.load_extension(
            os.path.join(_mod_spatialite_path, 'mod_spatialite')
        )
    else:
        print(
            "INFO: found unsupported operating system! mod_spatialite "
            "dependencies may not be loaded correctly."
        )
        dbapi_conn.load_extension(
            os.path.join(_mod_spatialite_path, 'mod_spatialite')
        )


def init_sqlite_engine(database_path, **kwargs):
    global _engine
    global Session
    global _mod_spatialite_path

    _mod_spatialite_path = kwargs.pop(
        'mod_spatialite_path',
        os.path.join(os.path.dirname(__file__), '..', 'mod_spatialite')
    )

    _engine = create_engine("sqlite:///%s" % database_path, **kwargs)
    # removes all existing sessions as they cannot be re-configurated once used
    Session.remove()
    Session.configure(bind=_engine)

    # set sqlite pragmas
    listen(_engine, 'connect', set_sqlite_pragma)

    # load spatialite extension
    listen(_engine, 'connect', load_spatialite)
    connection = _engine.connect()
    connection.execute(select([func.InitSpatialMetaData()]))
    connection.close()

    return _engine


Session = scoped_session(sessionmaker(bind=_engine))


class _Base(object):
    query = Session.query_property()

    def add(self):
        Session.add(self)
        Session.flush()


Base = declarative_base(cls=_Base)
