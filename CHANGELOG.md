# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.0.3]

### Added
- more verbose `grid run` errors
- `--pattern` argument replaces `--name` for a full path towards file replica
- `--force`, `--dry`, `--exec` arguments

### Removed
- `grid run` (replace by `grid new PATTERN(s) --exec COMMAND`)
  and `grid distribute` (replaced by `--force` arg)


## [0.0.2]

### Added
- create file `.variables` with all variables computed per grid folder
- formatting support


## [0.0.1]

### Added
- package
- `grid new -r` to find files and patterns recursively
- other command-line options

### Fixed
- switch to `file-grid` package name
- removed pyparsing and switched to python eval engine
- performance optimizations
- cleaner 'distribute' behavior
- project structure

### Removed
- optimization routines
