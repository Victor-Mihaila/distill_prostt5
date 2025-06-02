# Ablation Studies

* I performed something of an ablation study to determine optimal architecture (for a certain parameter count)
* Every model was run on a single MI250x on Pawsey with my `distill_prostt5_0.4.1.sif` container
* The run script was

```bash
BATCH_SIZE=32
WARMUP_RATIO="0.05"
python distill_prostt5/run.py train -p /mynvme/data/prostT5_training.h5 -e  /mynvme/data/prostT5_validation.h5  -o $OUTDIR --learning_rate 8e-4  --no_logits --warmup_ratio $WARMUP_RATIO  -a 1 --epochs 10 -b $BATCH_SIZE --logging_eval_steps 20000  --num_workers 8 --num_heads $NUM_HEADS --num_layers $NUM_LAYERS --hidden_size $HIDDEN_DIM --intermediate_size $INTERMEDIATE_DIM --save_steps 20000
```

* Across this sweep of layers, heads, and dimensions

```bash
for NUM_HEADS in 6 8 10 12 14 16; do
  # Hidden dim must be multiple of NUM_HEADS
  for HIDDEN_DIM in $(seq $((NUM_HEADS*8)) $((NUM_HEADS*8)) $((NUM_HEADS*64))); do
    # Intermediate dim can vary from HIDDEN_DIM/2 to 2*HIDDEN_DIM
    for INTERMEDIATE_DIM in $(seq $((HIDDEN_DIM / 2)) $((HIDDEN_DIM / 2)) $((2 * HIDDEN_DIM))); do
      # Ensure it's ≥ 1
      if [ "$INTERMEDIATE_DIM" -lt 1 ]; then continue; fi

      for NUM_LAYERS in $(seq $((NUM_HEADS + 2)) 2 $((NUM_HEADS + 12))); do
        # Create unique job name
        JOB_NAME="${JOB_PREFIX}_h${NUM_HEADS}_l${NUM_LAYERS}_d${HIDDEN_DIM}_i${INTERMEDIATE_DIM}"

        # Make a copy of base script
        SCRIPT_NAME="${JOB_NAME}.sh"
        cp "$BASE_SCRIPT" "$SCRIPT_NAME"

        # Replace or export updated variables in the new script
        sed -i "s/^NUM_HEADS=.*/NUM_HEADS=${NUM_HEADS}/" "$SCRIPT_NAME"
        sed -i "s/^NUM_LAYERS=.*/NUM_LAYERS=${NUM_LAYERS}/" "$SCRIPT_NAME"
        sed -i "s/^HIDDEN_DIM=.*/HIDDEN_DIM=${HIDDEN_DIM}/" "$SCRIPT_NAME"
        sed -i "s/^INTERMEDIATE_DIM=.*/INTERMEDIATE_DIM=${INTERMEDIATE_DIM}/" "$SCRIPT_NAME"
        #sed -i "s/^BATCH_SIZE=.*/BATCH_SIZE=${BATCH_SIZE}/" "$SCRIPT_NAME"
        #sed -i "s/^WARMUP_RATIO=.*/WARMUP_RATIO=${WARMUP_RATIO}/" "$SCRIPT_NAME"

        # Update output/err file names
        sed -i "s/^#SBATCH --job-name=.*/#SBATCH --job-name=${JOB_NAME}/" "$SCRIPT_NAME"
        sed -i "s/^#SBATCH --output=.*/#SBATCH --output=${JOB_NAME}-%j.log/" "$SCRIPT_NAME"
        sed -i "s/^#SBATCH --err=.*/#SBATCH --err=${JOB_NAME}-%j.err/" "$SCRIPT_NAME"

        # Submit the job
        echo "Submitting $SCRIPT_NAME"
        sbatch "$SCRIPT_NAME"
      done
    done
  done

``` 

* `ablation_params.tsv` contains the heads, layers, dimensions and parameter counts 
* Largest parameter count was `293708800`, ignored ones under 1M generally

## Loss curves

* The general code to get these as required

```bash
# Create the output directory if it doesn't exist
mkdir -p trainer_states

# Loop through all matching directories
for dir in *_*_*.*/; do
    # Extract parts from directory name
    if [[ $dir =~ ^([0-9]+)_([0-9]+)_([0-9]+)_single_gpu_([0-9]+)_([0-9]+)_from_scratch_.*\/$ ]]; then
        heads="${BASH_REMATCH[1]}"
        layers="${BASH_REMATCH[2]}"
        hidden_dim="${BASH_REMATCH[4]}"
        intermediate_dim="${BASH_REMATCH[5]}"

        # Find the checkpoint subdirectory
        for ckpt_dir in "$dir"/checkpoint-*; do
            if [[ -d "$ckpt_dir" ]]; then
                checkpoint=$(basename "$ckpt_dir" | cut -d'-' -f2)
                src="$ckpt_dir/trainer_state.json"
                if [[ -f "$src" ]]; then
                    dst="trainer_states/${heads}_${layers}_${hidden_dim}_${intermediate_dim}_checkpoint-${checkpoint}.json"
                    cp "$src" "$dst"
                    echo "Copied to $dst"
                fi
            fi
        done
    fi
done
```

* Then to analyse - Juypter notebook

# Takeaways

* Parameter count was easily the most important thing
* Compressed intermediate dimension generally performed worse than not compressed unsurpisingly
* For a given parameter count, generally a model with more layers compared to attention heads performed the best (echoing the modernbert papers)

* Choose to train 5M, 10M, 15M, 25M and 50M models to completion going forward based on the ablation data
* Based on the training as of 2 June 2025, I chose:

5M: 10_22_160_240 (based on lowest loss after 1.5M steps)
10M: 10_20_240_360 (loss after 1M)
15M: 16_26_256_384 (2nd lowest loss after 800k - with no compression/long layers)
25M: 10_22_320_640 (lowest loss at 640k)
50M: 14_24_448_896 (lowest loss at 500k)

* Then also modernBERT 12_22_768_1152 (110M), 4_24_896_1536 (176M) modernBERT large 16_28_1024_2624 (345M) and 30_40_1500_3600 (1B)



