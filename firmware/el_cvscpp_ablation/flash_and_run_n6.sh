#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ENV_FILE=${EL_CVSCPP_ENV_FILE:-"$SCRIPT_DIR/.env"}
LOAD_ENV="1"

index=1
while [ "$index" -le "$#" ]; do
  eval "arg=\${$index}"
  case "$arg" in
    --env-file)
      index=$((index + 1))
      [ "$index" -le "$#" ] || {
        printf 'ERROR: --env-file requires a value\n' >&2
        exit 1
      }
      eval "ENV_FILE=\${$index}"
      ;;
    --no-env-file)
      LOAD_ENV="0"
      ;;
  esac
  index=$((index + 1))
done

if [ "$LOAD_ENV" = "1" ] && [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

PROJECT_ROOT=${EL_CVSCPP_PROJECT_ROOT:-}
EDGE_C_ROOT=${EL_CVSCPP_EDGE_C_ROOT:-}
RLTOOLS_ROOT=${EL_CVSCPP_RLTOOLS_ROOT:-}
TOOLCHAIN_BIN=${EL_CVSCPP_TOOLCHAIN_BIN:-}
STLINK_GDBSERVER=${EL_CVSCPP_STLINK_GDBSERVER:-}
CUBEPROG_BIN=${EL_CVSCPP_CUBEPROG_BIN:-}
EXTERNAL_LOADER=${EL_CVSCPP_EXTERNAL_LOADER:-}
PYTHON=${EL_CVSCPP_PYTHON:-python3}
APPLI_PROJECT_NAME=${EL_CVSCPP_APPLI_PROJECT_NAME:-EL_C_vsCpp_Appli}
FSBL_PROJECT_NAME=${EL_CVSCPP_FSBL_PROJECT_NAME:-EL_C_vsCpp_FSBL}
BUILD_CONFIG=${EL_CVSCPP_BUILD_CONFIG:-EL_C_vsCpp_Ablation_DEBUG}
SERIAL_PORT=${EL_CVSCPP_SERIAL_PORT:-}
SERIAL_BAUDRATE=${EL_CVSCPP_SERIAL_BAUDRATE:-12000000}
SERIAL_TIMEOUT=${EL_CVSCPP_SERIAL_TIMEOUT:-120}
GDB_PORT=${EL_CVSCPP_GDB_PORT:-61234}
JOBS=${EL_CVSCPP_JOBS:-7}
INPUT_FEATURES=${EL_CVSCPP_INPUT_FEATURES:-3}
H1="8"
H2="8"
VARIANT="cpp_m55"
SKIP_BUILD="0"
SKIP_FLASH="0"
SKIP_FSBL="0"
SKIP_CAPTURE="0"
DRY_RUN="0"
CAPTURE_PID=""
GDBSERVER_PID=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Build, flash, boot, and capture one EL_C_vsCpp ablation firmware case.

Configuration is read from firmware/el_cvscpp_ablation/.env by default. Copy
.env.example to .env and adjust the local paths. Environment variables may also
be set directly; CLI options override both.

Options:
  --env-file PATH         Load a shell-style .env file. Default: $ENV_FILE
  --no-env-file           Do not load a .env file.
  --input-features N      Input feature count. Default: $INPUT_FEATURES
  --config H1xH2          Hidden sizes, e.g. 64x32.
  --variant NAME          all, legacy_c, cpp_direct_c_backend, cpp_m55, cpp_generic, or rltools_generic.
                          Default: $VARIANT
  --h1 N                  First hidden width. Default: 8.
  --h2 N                  Second hidden width. Default: 8.
  --serial-port PATH      UART device used for capture.
  --serial-baudrate N     Default: $SERIAL_BAUDRATE
  --serial-timeout SEC    Default: $SERIAL_TIMEOUT
  --project-root PATH     CubeIDE EL_C_vsCpp project root.
  --edge-c-root PATH      External legacy EdgeLearning checkout. Required only for all, legacy_c,
                          and cpp_direct_c_backend.
  --rltools-root PATH     External RLTools checkout.
  --toolchain-bin PATH    Directory containing arm-none-eabi tools.
  --stlink-gdbserver PATH ST-LINK_gdbserver executable.
  --cubeprog-bin PATH     CubeProgrammer bin directory.
  --external-loader PATH  STM32N6 external-loader .stldr.
  --python PATH           Python with pyserial. Default: $PYTHON
  --appli-project-name N  Default: $APPLI_PROJECT_NAME
  --fsbl-project-name N   Default: $FSBL_PROJECT_NAME
  --build-config N        Default: $BUILD_CONFIG
  --gdb-port N            Default: $GDB_PORT
  --jobs N                Default: $JOBS
  --skip-build            Reuse existing app ELF.
  --skip-flash            Skip app flash.
  --skip-fsbl             Do not launch FSBL.
  --skip-capture          Do not capture UART output.
  --dry-run               Print commands only.
EOF
}

log() {
  printf '%s\n' "$*"
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_arg() {
  [ "$#" -ge 2 ] || fail "Option $1 requires a value."
}

variant_requires_legacy_c() {
  case "$1" in
    all|legacy_c|cpp_direct_c_backend) return 0 ;;
    *) return 1 ;;
  esac
}

while [ $# -gt 0 ]; do
  case "$1" in
    --env-file)
      require_arg "$@"
      ENV_FILE=$2
      shift 2
      ;;
    --no-env-file)
      LOAD_ENV="0"
      shift 1
      ;;
    --input-features) require_arg "$@"; INPUT_FEATURES=$2; shift 2 ;;
    --config)
      require_arg "$@"
      case "$2" in
        *x*) H1=${2%x*}; H2=${2#*x} ;;
        *) fail "--config must use H1xH2, e.g. 64x32" ;;
      esac
      shift 2
      ;;
    --variant) require_arg "$@"; VARIANT=$2; shift 2 ;;
    --h1) require_arg "$@"; H1=$2; shift 2 ;;
    --h2) require_arg "$@"; H2=$2; shift 2 ;;
    --serial-port) require_arg "$@"; SERIAL_PORT=$2; shift 2 ;;
    --serial-baudrate) require_arg "$@"; SERIAL_BAUDRATE=$2; shift 2 ;;
    --serial-timeout) require_arg "$@"; SERIAL_TIMEOUT=$2; shift 2 ;;
    --project-root) require_arg "$@"; PROJECT_ROOT=$2; shift 2 ;;
    --edge-c-root) require_arg "$@"; EDGE_C_ROOT=$2; shift 2 ;;
    --rltools-root) require_arg "$@"; RLTOOLS_ROOT=$2; shift 2 ;;
    --toolchain-bin) require_arg "$@"; TOOLCHAIN_BIN=$2; shift 2 ;;
    --stlink-gdbserver) require_arg "$@"; STLINK_GDBSERVER=$2; shift 2 ;;
    --cubeprog-bin) require_arg "$@"; CUBEPROG_BIN=$2; shift 2 ;;
    --external-loader) require_arg "$@"; EXTERNAL_LOADER=$2; shift 2 ;;
    --python) require_arg "$@"; PYTHON=$2; shift 2 ;;
    --appli-project-name) require_arg "$@"; APPLI_PROJECT_NAME=$2; shift 2 ;;
    --fsbl-project-name) require_arg "$@"; FSBL_PROJECT_NAME=$2; shift 2 ;;
    --build-config) require_arg "$@"; BUILD_CONFIG=$2; shift 2 ;;
    --gdb-port) require_arg "$@"; GDB_PORT=$2; shift 2 ;;
    --jobs) require_arg "$@"; JOBS=$2; shift 2 ;;
    --skip-build) SKIP_BUILD="1"; shift 1 ;;
    --skip-flash) SKIP_FLASH="1"; shift 1 ;;
    --skip-fsbl) SKIP_FSBL="1"; shift 1 ;;
    --skip-capture) SKIP_CAPTURE="1"; shift 1 ;;
    --dry-run) DRY_RUN="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

[ -n "$PROJECT_ROOT" ] || fail "EL_CVSCPP_PROJECT_ROOT is required in $ENV_FILE or via --project-root"
[ -n "$TOOLCHAIN_BIN" ] || fail "EL_CVSCPP_TOOLCHAIN_BIN is required in $ENV_FILE or via --toolchain-bin"
[ -n "$STLINK_GDBSERVER" ] || fail "EL_CVSCPP_STLINK_GDBSERVER is required in $ENV_FILE or via --stlink-gdbserver"
[ -n "$CUBEPROG_BIN" ] || fail "EL_CVSCPP_CUBEPROG_BIN is required in $ENV_FILE or via --cubeprog-bin"

case "$VARIANT" in
  all|legacy_c|cpp_direct_c_backend|cpp_m55|cpp_generic|rltools_generic) ;;
  *) fail "--variant must be one of: all legacy_c cpp_direct_c_backend cpp_m55 cpp_generic rltools_generic" ;;
esac

if variant_requires_legacy_c "$VARIANT"; then
  [ -n "$EDGE_C_ROOT" ] || fail "EL_CVSCPP_EDGE_C_ROOT is required in $ENV_FILE or via --edge-c-root for variant $VARIANT"
  [ -f "$EDGE_C_ROOT/Inc/edgelearning.h" ] || fail "Legacy C header not found: $EDGE_C_ROOT/Inc/edgelearning.h"
fi

case "$VARIANT" in
  all|rltools_generic)
    [ -n "$RLTOOLS_ROOT" ] || fail "EL_CVSCPP_RLTOOLS_ROOT is required in $ENV_FILE or via --rltools-root for variant $VARIANT"
    [ -d "$RLTOOLS_ROOT/include/rl_tools" ] || fail "RLTools include directory not found: $RLTOOLS_ROOT/include/rl_tools"
    ;;
esac

if [ -z "$EXTERNAL_LOADER" ]; then
  EXTERNAL_LOADER="$CUBEPROG_BIN/ExternalLoader/MX25UM51245G_STM32N6570-NUCLEO.stldr"
fi

CONFIG_SUFFIX="${INPUT_FEATURES}_${H1}x${H2}_1"
if [ "$VARIANT" = "all" ]; then
  APP_BUILD_DIR="$PROJECT_ROOT/STM32CubeIDE/Appli/$BUILD_CONFIG/$CONFIG_SUFFIX"
  APP_ELF="$APP_BUILD_DIR/${APPLI_PROJECT_NAME}_$CONFIG_SUFFIX.elf"
else
  APP_BUILD_DIR="$PROJECT_ROOT/STM32CubeIDE/Appli/$BUILD_CONFIG/$CONFIG_SUFFIX/$VARIANT"
  APP_ELF="$APP_BUILD_DIR/${APPLI_PROJECT_NAME}_${CONFIG_SUFFIX}_${VARIANT}.elf"
fi
FSBL_ELF="$PROJECT_ROOT/STM32CubeIDE/FSBL/Debug/$FSBL_PROJECT_NAME.elf"
CAPTURE_LOG="$APP_BUILD_DIR/serial.log"

[ -x "$STLINK_GDBSERVER" ] || fail "ST-LINK_gdbserver not executable: $STLINK_GDBSERVER"
[ -x "$TOOLCHAIN_BIN/arm-none-eabi-gdb" ] || fail "arm-none-eabi-gdb not found in $TOOLCHAIN_BIN"
[ -d "$CUBEPROG_BIN" ] || fail "CubeProgrammer bin not found: $CUBEPROG_BIN"
[ -f "$EXTERNAL_LOADER" ] || fail "External loader not found: $EXTERNAL_LOADER"
[ -x "$PYTHON" ] || fail "Python not executable: $PYTHON"
[ "$SKIP_FSBL" = "1" ] || [ -f "$FSBL_ELF" ] || fail "FSBL ELF not found: $FSBL_ELF"
if [ "$SKIP_CAPTURE" != "1" ]; then
  [ -n "$SERIAL_PORT" ] || fail "EL_CVSCPP_SERIAL_PORT is required in $ENV_FILE or via --serial-port"
fi

cleanup() {
  if [ -n "$GDBSERVER_PID" ] && kill -0 "$GDBSERVER_PID" 2>/dev/null; then
    kill "$GDBSERVER_PID" 2>/dev/null || true
    wait "$GDBSERVER_PID" 2>/dev/null || true
  fi
  if [ -n "$CAPTURE_PID" ] && kill -0 "$CAPTURE_PID" 2>/dev/null; then
    kill "$CAPTURE_PID" 2>/dev/null || true
    wait "$CAPTURE_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM HUP

run_cmd() {
  log "\$ $*"
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

start_gdbserver() {
  mode=$1
  if [ "$mode" = "fsbl" ]; then
    log "\$ $STLINK_GDBSERVER -p $GDB_PORT -l 1 -d -z 61335 -s -cp $CUBEPROG_BIN -m 1 -k"
    if [ "$DRY_RUN" = "1" ]; then
      return 0
    fi
    "$STLINK_GDBSERVER" -p "$GDB_PORT" -l 1 -d -z 61335 -s -cp "$CUBEPROG_BIN" -m 1 -k &
  else
    log "\$ $STLINK_GDBSERVER -p $GDB_PORT -l 1 -d -s -cp $CUBEPROG_BIN -el $EXTERNAL_LOADER -m 1 -k"
    if [ "$DRY_RUN" = "1" ]; then
      return 0
    fi
    "$STLINK_GDBSERVER" -p "$GDB_PORT" -l 1 -d -s -cp "$CUBEPROG_BIN" -el "$EXTERNAL_LOADER" -m 1 -k &
  fi
  GDBSERVER_PID=$!
  sleep 1.5
  kill -0 "$GDBSERVER_PID" 2>/dev/null || fail "ST-LINK_gdbserver exited early"
}

stop_gdbserver() {
  if [ -n "$GDBSERVER_PID" ] && kill -0 "$GDBSERVER_PID" 2>/dev/null; then
    kill "$GDBSERVER_PID" 2>/dev/null || true
    wait "$GDBSERVER_PID" 2>/dev/null || true
  fi
  GDBSERVER_PID=""
}

gdb_load_app() {
  start_gdbserver app
  run_cmd "$TOOLCHAIN_BIN/arm-none-eabi-gdb" --batch -q "$APP_ELF" \
    -ex "target remote localhost:$GDB_PORT" \
    -ex "monitor reset" \
    -ex "load" \
    -ex "quit"
  stop_gdbserver
}

gdb_load_fsbl() {
  start_gdbserver fsbl
  run_cmd "$TOOLCHAIN_BIN/arm-none-eabi-gdb" --batch -q "$FSBL_ELF" \
    -ex "target remote localhost:$GDB_PORT" \
    -ex "monitor reset" \
    -ex "load" \
    -ex "detach" \
    -ex "quit"
  stop_gdbserver
}

log "env_file=$ENV_FILE"
log "config=${INPUT_FEATURES}-${H1}x${H2}-1"
log "variant=$VARIANT"
log "edge_c_root=$EDGE_C_ROOT"
log "rltools_root=$RLTOOLS_ROOT"
log "app_elf=$APP_ELF"
log "fsbl_elf=$FSBL_ELF"
log "serial_port=$SERIAL_PORT"
log "capture_log=$CAPTURE_LOG"

if [ "$SKIP_BUILD" != "1" ]; then
  run_cmd make -s -f "$SCRIPT_DIR/Makefile" -j"$JOBS" \
    H1="$H1" \
    H2="$H2" \
    INPUT_FEATURES="$INPUT_FEATURES" \
    PROJECT_ROOT="$PROJECT_ROOT" \
    EDGE_C_ROOT="$EDGE_C_ROOT" \
    RLTOOLS_ROOT="$RLTOOLS_ROOT" \
    TOOLCHAIN_BIN="$TOOLCHAIN_BIN" \
    APPLI_PROJECT_NAME="$APPLI_PROJECT_NAME" \
    BUILD_CONFIG="$BUILD_CONFIG" \
    VARIANT="$VARIANT"
fi

[ -f "$APP_ELF" ] || fail "Application ELF not found: $APP_ELF"

if [ "$SKIP_FLASH" != "1" ]; then
  gdb_load_app
fi

if [ "$DRY_RUN" != "1" ] && [ "$SKIP_CAPTURE" != "1" ]; then
  "$PYTHON" "$SCRIPT_DIR/serial_capture.py" \
    --port "$SERIAL_PORT" \
    --baudrate "$SERIAL_BAUDRATE" \
    --timeout "$SERIAL_TIMEOUT" \
    --output "$CAPTURE_LOG" &
  CAPTURE_PID=$!
  sleep 0.5
fi

if [ "$SKIP_FSBL" != "1" ]; then
  gdb_load_fsbl
fi

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

if [ "$SKIP_CAPTURE" = "1" ]; then
  exit 0
fi

if [ -n "$CAPTURE_PID" ]; then
  wait "$CAPTURE_PID"
  CAPTURE_STATUS=$?
  CAPTURE_PID=""
  exit "$CAPTURE_STATUS"
fi

exit 0
