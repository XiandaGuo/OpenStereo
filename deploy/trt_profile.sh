#!/bin/bash

# init vars
ENGINE_FILE=""
ONNX_FILE=""
LOG_FILE=""
FP16=false
VERBOSE=false
SAVE_ENGINE=""

# help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --loadEngine PATH    Load TensorRT engine file from PATH"
    echo "  --onnx PATH          Load ONNX model file from PATH"
    echo "  --saveEngine PATH    Save TensorRT engine file to PATH"
    echo "  --log PATH           Save output log to PATH"
    echo "  --fp16               Use FP16 precision"
    echo "  -v, --verbose        Enable verbose output"
    echo "  -h, --help           Display this help and exit"
}

# parse options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --loadEngine)
            if [[ -n "$2" && "$2" != --* ]]; then
                ENGINE_FILE="$2"
                shift
            else
                echo "Error: --loadEngine flag requires an engine file path."
                exit 1
            fi
            ;;
        --onnx)
            if [[ -n "$2" && "$2" != --* ]]; then
                ONNX_FILE="$2"
                shift
            else
                echo "Error: --onnx flag requires an ONNX file path."
                exit 1
            fi
            ;;
        --saveEngine)
            if [[ -n "$2" && "$2" != --* ]]; then
                SAVE_ENGINE="$2"
                shift
            else
                echo "Error: --saveEngine flag requires an engine file path."
                exit 1
            fi
            ;;
        --log)
            if [[ -n "$2" && "$2" != --* ]]; then
                LOG_FILE="$2"
                shift
            else
                echo "Error: --log flag requires a log file path."
                exit 1
            fi
            ;;
        --fp16)
            FP16=true
            ;;
        -v|--verbose)
            VERBOSE=true
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown flag '$1'"
            show_help
            exit 1
            ;;
    esac
    shift
done

# check ENGINE_FILE and ONNX_FILE
if [[ -z "$ENGINE_FILE" && -z "$ONNX_FILE" ]]; then
    echo "Error: Either --loadEngine or --onnx flag must be specified."
    show_help
    exit 1
fi

# build trtexec instruction
TRTEXEC_CMD="trtexec"
if [[ -n "$ENGINE_FILE" ]]; then
    TRTEXEC_CMD+=" --loadEngine=$ENGINE_FILE"
elif [[ -n "$ONNX_FILE" ]]; then
    TRTEXEC_CMD+=" --onnx=$ONNX_FILE"
fi

if [[ -n "$SAVE_ENGINE" ]]; then
    TRTEXEC_CMD+=" --saveEngine=$SAVE_ENGINE"
fi

TRTEXEC_CMD+=" --dumpProfile --profilingVerbosity=detailed"

if [[ "$FP16" = true ]]; then
    TRTEXEC_CMD+=" --fp16"
fi

if [[ "$VERBOSE" = true ]]; then
    TRTEXEC_CMD+=" --verbose"
fi

if [[ -n "$LOG_FILE" ]]; then
    $TRTEXEC_CMD > $LOG_FILE 2>&1
else
    $TRTEXEC_CMD
fi

# check trtexec result
if [ $? -eq 0 ]; then
    if [[ -n "$LOG_FILE" ]]; then
        echo "Profile completed successfully. Output saved to '$LOG_FILE'."
    else
        echo "Profile completed successfully."
    fi
else
    if [[ -n "$LOG_FILE" ]]; then
        echo "Error: trtexec failed. Check '$LOG_FILE' for details."
    else
        echo "Error: trtexec failed."
    fi
    exit 1
fi
