Reservoir Stuff


tail -f churro_job.log

sacct -u charanc2

squeue -u charanc2

sacct -j 425896 --format=JobID,JobName,Start,End,Elapsed

rsync -avz --exclude='.git' --filter=':- ./home/charanc2/projects/Reservoir-Research/.gitignore' charanc2@lakeshore.acer.uic.edu:/home/charanc2/projects/Reservoir-Research/ /Users/churroc/Personal/code/reservoir_research/
-a is archive, -v is verbose, -z is compression


rsync -avz --exclude='.git' --filter=':- /home/charanc2/projects/Reservoir-Research/.gitignore' charanc2@lakeshore.acer.uic.edu:/home/charanc2/projects/Reservoir-Research/src/week7/1_test_if_working/bayesian_job /Users/churroc/Personal/code/reservoir_research/src/week7_8/1_test_if_working/bayesian_job


This one is syncing local to the cluster
rsync -avz --exclude='.git' --filter=':- /Users/churroc/Personal/code/reservoir_research/.gitignore' /Users/churroc/Personal/code/reservoir_research/ charanc2@lakeshore.acer.uic.edu:/home/charanc2/projects/Reservoir-Research/


TODO
Hyseria
Do GPU stuff
instead of havr rest lenth between 0 to 2 have instead be 1 to 2 since 0 to 1 can stop it