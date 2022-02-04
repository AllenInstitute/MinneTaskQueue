export USER=forrestc
gcloud config set project em-270621

gcloud beta container --project "em-270621" clusters create "${USER}-cluster-2" --zone "us-east1-b" --no-enable-basic-auth --cluster-version "1.21.5-gke.1302" --release-channel "regular" --machine-type "e2-standard-8" --image-type "COS_CONTAINERD" --disk-type "pd-standard" --disk-size "100" --metadata disable-legacy-endpoints=true --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --max-pods-per-node "110" --preemptible --num-nodes "1" --logging=SYSTEM,WORKLOAD --monitoring=SYSTEM --enable-ip-alias --network "projects/em-270621/global/networks/default" --subnetwork "projects/em-270621/regions/us-east1/subnetworks/default" --no-enable-intra-node-visibility --default-max-pods-per-node "110" --enable-autoscaling --min-nodes "0" --max-nodes "40" --no-enable-master-authorized-networks --addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver --enable-autoupgrade --enable-autorepair --max-surge-upgrade 1 --max-unavailable-upgrade 0 --enable-shielded-nodes --node-locations "us-east1-b"



gcloud container clusters get-credentials --zone us-east1-b ${USER}-cluster-2

kubectl create secret generic secretsminnie \
--from-file=google-secret.json=$HOME/.cloudvolume/secrets/v1dd-pcg-google-secret.json \
--from-file=$HOME/.cloudvolume/secrets/aws-secret.json \
--from-file=cave-secret.json=cave-token.json

kubectl create secret generic boto \
--from-file=minimal_boto 

kubectl create secret generic secrets3 \
--from-file=google-secret.json=$HOME/.cloudvolume/secrets/v1dd-pcg-google-secret.json \
--from-file=$HOME/.cloudvolume/secrets/aws-secret.json \
--from-file=cave-secret.json=$HOME/.cloudvolume/secrets/minnie.microns-daf.com-cave-secret.json

