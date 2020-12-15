export USER=forrestc
gcloud config set project em-270621

gcloud beta container --project "em-270621" clusters create "${USER}-cpu-cluster-1" --zone "us-east1-b" --no-enable-basic-auth --cluster-version "1.15.12-gke.2" --machine-type "n1-standard-1" --image-type "COS" --disk-type "pd-standard" --disk-size "100" --metadata disable-legacy-endpoints=true --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --num-nodes "1" --enable-stackdriver-kubernetes --enable-ip-alias --network "projects/em-270621/global/networks/default" --subnetwork "projects/em-270621/regions/us-east1/subnetworks/default" --default-max-pods-per-node "110" --no-enable-master-authorized-networks --addons HorizontalPodAutoscaling,HttpLoadBalancing --enable-autoupgrade --enable-autorepair && gcloud beta container --project "em-270621" node-pools create "cpu-pool-1" --cluster "${USER}-cpu-cluster-1" --zone "us-east1-b" --node-version "1.15.12-gke.2" --machine-type "n1-standard-8" --image-type "COS" --disk-type "pd-standard" --disk-size "100" --metadata disable-legacy-endpoints=true --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --preemptible --enable-autoscaling --min-nodes "0" --max-nodes "16"  --enable-autoupgrade --enable-autorepair


gcloud container clusters get-credentials --zone us-east1-b ${USER}-cpu-cluster-1


kubectl create secret generic secrets2 \
--from-file=$HOME/.cloudvolume/secrets/google-secret.json \
--from-file=$HOME/.cloudvolume/secrets/aws-secret.json \
--from-file=chunkedgraph-secret.json=./allen_token.json

kubectl create secret generic boto \
--from-file=minimal_boto 