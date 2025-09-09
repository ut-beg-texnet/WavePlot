      MaxCrackD    =   100;
      CrackDMapRec = RECORD
                      Offset        : LongInt;
                      {Values passed from WAVE}
                         Time          : Single;
                         ic1,ic2       : LongInt;
                         gnum, gID     : LongInt;  {ID of geometry}
                         VQty, StQty   : LongInt;  {Qty of crack Vars and stopes}
                         CrLen         : LongInt;  {Length, entry in crack file}
                    END; {CrackMapRec}
      CrackRec    = RECORD
                      {Values passed from WAVE}
                      vNs1,vNs0,vT1s1,vT1s0 : single;
                      uNs1,uNs0,uT1s1,uT1s0 : single;
                      sNs1,sNs0,sT1s1,sT1s0 : single;
                    END;
      CrackDMapArray = ARRAY [1..MaxCrackD] OF CrackDMapRec;
      CrackFile    : File;
      CrackDMap    : ^CrackDMapArray;
      NoSnap,NoDump,NoHist,NoGeom,NoCrackD              : Boolean;
      QtySnap,QtyHist,QtyDump,QtyGeom,QtyTraj,QtyCrackD : Word;

                    ASSIGN (CrackFile ,    (Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.CRK') );
                    NO_error := NO_error AND (IOResult = 0);
                    RESET  (CrackFile);
                    NoCrackD := (IOResult <> 0);


    PROCEDURE DI_ReadMap;
      VAR
        SnapStart,DumpStart,HistStart,GeomStart,CrackDStart : LongInt;
        DumInt    : LongInt;
        PlType    : LongInt;
        i         : Integer;
      BEGIN
        QtySnap:=0; QtyDump:=0; QtyHist:=0; QtyGeom:=0; QtyCrackD:=0;
        SnapStart:=0; DumpStart:=0; HistStart:=0; GeomStart:=0; CrackDStart:=0;
        WHILE (NOT (EOF(MapFile)) ) DO
          BEGIN

              4 : BEGIN
                    IF (QtyCrackD<MaxCrackD) THEN Inc(QtyCrackD);
                    WITH CrackDMap^[QtyCrackD] DO
                      BEGIN
                        BlockRead (MapFile, Time, 32);
                        Seek (MapFile, FilePos(MapFile)+88);
                        Offset      := CrackDStart;
                        CrackDStart := CrackDStart+CrLen;
                      END;
                  END;
            END; {Case}
          END; {While}

      END; {DI_ReadMap}

  PROCEDURE Ds_GraphCrackD (D : Word);
    TYPE
      CrackArray = ARRAY [1..1000] OF CrackRec;
{      CrackRec    = RECORD
                      vNs1,vNs0,vT1s1,vT1s0 : single;
                      uNs1,uNs0,uT1s1,uT1s0 : single;
                      sNs1,sNs0,sT1s1,sT1s0 : single;
                    END;}
    VAR
      st,iqty,CrPos,isz : LongInt;
      Cracks : ^CrackArray;
      CrackF : File;
    BEGIN {Ds_GraphCrackD}
      st := CrackDMap^[D].ic1;
      CrPos := CrackDMap^[D].Offset;
      ASSIGN (CrackF, (Sets.Drive+':'+Sets.Dir+'\'+Sets.FName+'.CRK') );
      RESET  (CrackF, 4);

      REPEAT
        Seek (CrackF, CrPos+1);
        BlockRead (CrackF,st,1*4);
        BlockRead (CrackF,iqty,1*4);
        CrPos := CrPos+1;
        isz := iqty*CrackDMap^[D].Vqty;
        GetMem (Cracks,isz*4);
        Seek (CrackF, CrPos+1);
        BlockRead (CrackF,Cracks^,isz);

        WITH Stopes^[st] DO
          BEGIN

            {read current crack}

            {graph crack}
{            IF (Inview (Dir, Slice, i1,i2, j1,j2, k1,k2)) THEN
              BEGIN
                IF (stype IN [2] )
                   THEN BEGIN  SetColor (LightRed); SetLineStyle (SolidLn,0,Thickwidth);  END
                   ELSE BEGIN  Setcolor (LightRed);SetLineStyle (SolidLn,0,Normwidth);  END;
                IF (Layout=Nil)
                   THEN CASE Dir OF
                          0 : Draw2D_Rect (i1,j1,i2,j2);
                          1 : Draw2D_Rect (i1,k1,i2,k2);
                          2 : Draw2D_Rect (k1,j1,k2,j2);
                        END; {Case}

            FreeMem (Cracks,isz*4);


          END; {With}


      UNTIL (st=CrackDMap^[D].ic2);

    END; {Ds_GraphGeom}



c
      subroutine dumpcrack (npar)
c    -- dumps all vels, displs, stresses on given cracks --
      logical chkint
      include 'wavcom.inc'
      include 'stocom.inc'
      include 'matcom.inc'
      include 'varcom.inc'
      include 'lstcom.inc'
      character*1 key(5)
      data key
     . /'r','e','p',' ',
     . iterm/
c    -- get range of cracks --
      if (chkint(npar,ic1)) then
         ic1=0
        else
           npar=npar+1
           if (chkint(npar,ic2)) then
              ic2=ic1
             else
              npar=npar+1
           endif
      endif
c    -- find first and last crack positions --
      is2=0
      if (ic1.eq.0) then
         is1=istpnt
        else
         is  = istpnt
         do while ((is.ne.0).and.(is2.eq.0))
            if (ia(is+kstid).eq.ic1) then
               if (is1.eq.0) then
                  is1 = is
                 else
                  is2 = is
               endif
            endif
            if (ia(is+kstid).eq.ic2) then
               if (is1.eq.0) then
                  is1 = is
                 else
                  is2 = is
               endif
            endif
            is = ia(is)
         enddo
      endif
c    -- if request is for a single crack dump, produce it --
      call xlocate (key(1),npar)
      if (miss) then
         call writedcrack (ic1,ic2,is1,is2)
       else
c       -- if repeated, add to the repeat list --
c       --   (check for change in repeat frequency, default to 1) --
         if (chkint(npar+1,idt)) then
              if (idcrpnt .eq. 0) idcrdt=1
c                otherwise leave unchanged
            else
              idcrdt = max0 (idt,1)
         endif
         write (wline, 2000) idcrdt
         call wr_line (yeslog)
c       --   add record to repeat list --
         call find (nwdcr,ip)
         if (error) return
         if (idcrpnt .eq. 0) idcrpnt = ip
         if (idcrend .ne. 0) ia(idcrend) = ip
         idmpend         = ip
         ia(ip)          = 0
         ia(ip+kdcric1)   = ic1
         ia(ip+kdcric2)   = ic2
         ia(ip+kdcris1)   = is1
         ia(ip+kdcris2)   = is2
      endif
      return
 2000 format ('** crack dump repeat frequency = ',i4,' cycles **')
      end
c
c
      subroutine writedcrack (ic1,ic2,is1,is2)
c    -- writes all vels, displs, stresses on given cracks to file --
      include 'wavcom.inc'
      include 'stocom.inc'
      include 'ppcom.inc'
      idum  = 0
      idumm = 120
      idumd = 0
c
      if (.not. ploflg) then
          call ppinit
      endif
      if (geomchanged) call plgeom
c
c    -- create entry in map file --
c    -- 7 4-byte numbers, pad to 128-byte record.
        nvqty = 18
        write (lunmap) idumm,
     .                 idcrack, time, ic1, ic2, nvqty, ignum, igid,
     .                 (idum, i=1,23),  idumm
c
c    -- write to crack file --
      is  = is1
      do while (is.ne.0)
         nid  = ia(is+kstid)
         call setstvar (is)
         im1 = (is+nwsthd)
         iqty = (iT1b-iT1a+1)*(iT2b-iT2a+1)
         idumd = 2*4
         write (luncrack) idumd,nid,iqty,idumd
c
         if (mode2d) then
            idumd = nvqty*4
            im  = im1
            do iT1 = iT1a,iT1b
               ig1   = ibase-1 + (iT1-1)*kT1stp + (iNa-1)*kNstp
               igT1n = ig1 + kT1stp
               write (luncrack) idumd,
     .           a(im+kvNr),a(im+kvNr),a(im+kvT1s0),a(im+kvT1s1),
     .           a(im+kuNr),a(im+kuNr),a(im+kvT1s0),a(im+kvT1s1),
     .           a(ig1+ksN),a(im+ksT1s0),a(im+ksT1s1),a(im+kshT1st),
     .           a(im+kT1slp),a(im+kTaslp),a(im+kNbnd),a(im+kShbnd),
     .           idumd
c               write (luncrack) idumd,nid,iT1,
c     .           a(im+kvNs0),a(im+kvNs1),a(im+kvT1s0),a(im+kvT1s1),
c     .           a(im+kuNs0),a(im+kuNs1),a(im+kuT1s0),a(im+kuT1s1),
c     .           a(ig1+ksN),a(im+ksT1s0),a(im+ksT1s1),a(im+kshT1st),
c     .           a(im+kT1slp),a(im+kTaslp),a(im+kNbnd),a(im+kShbnd),
c     .           idumd
               im = im + imT1stp
            enddo
         endif
c
         if (mode3d) then
           idumd = nvqty*4
           do iT2 = iT2a,iT2b
             im  = im1
             do iT1 = iT1a,iT1b
               ig1    = ibase-1 + (iT1-1)*kT1stp + (iT2-1)*kT2stp
     .                                           + (iNa-1)*kNstp
               igT1n = ig1 + kT1stp
               write (luncrack) idumd,nid,iT1,
     .           a(im+kvNr),a(im+kvNr),a(im+kvT1s0),a(im+kvT1s1),
     .           a(im+kuNr),a(im+kuNr),a(im+kvT1s0),a(im+kvT1s1),
     .           a(ig1+ksN),a(im+ksT1s0),a(im+ksT1s1),a(im+kshT1st),
     .           a(im+kT1slp),a(im+kTaslp),a(im+kNbnd),a(im+kShbnd),
     .           idumd
               im = im + imT1stp
             enddo
             im1 = im1 + imT2stp
           enddo
         endif
c
         if (is.eq.is2) then
            is=0
           else
            is = ia(is)
         endif
      enddo
c
      return
      end
c

c
      subroutine dcrrun
c    -- produce crack dumps from repeat list --
      include 'wavcom.inc'
      include 'lstcom.inc'
      ip = idcrpnt
      do while (ip .ne. 0)
         ic1    = ia(ip+kdcric1)
         ic2    = ia(ip+kdcric2)
         is1    = ia(ip+kdcris1)
         is2    = ia(ip+kdcris2)
         call writedcrack (ic1,ic2,is1,is2)
         ip = ia(ip)
      enddo
      return
      end
c
