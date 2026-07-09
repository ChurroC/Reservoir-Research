Reservoir Stuff


tail -f churro_job.log

sacct -u charanc2

squeue -u charanc2

sacct -j 425896 --format=JobID,JobName,Start,End,Elapsed

rsync -avz --filter=':- ./home/charanc2/projects/Reservoir-Research/.gitignore' charanc2@lakeshore.acer.uic.edu:/home/charanc2/projects/Reservoir-Research/ /Users/churroc/Personal/code/reservoir_research/
-a is archive, -v is verbose, -z is compression